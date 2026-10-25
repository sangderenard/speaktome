"""Read a Windows CPython 3.11 stack without injecting, suspending, or stopping it.

Uses the installed CPython 3.11 x64 structure layout. Only code locations and
selected compiler progress counters are printed. No third-party dependencies.
"""
import ctypes as c
import struct
import sys


def main(pid):
    if sys.version_info[:2] != (3, 11) or c.sizeof(c.c_void_p) != 8 or sys.platform != 'win32':
        raise RuntimeError('Requires Windows CPython 3.11 x64, matching the target')
    kernel = c.WinDLL('kernel32', use_last_error=True)
    psapi = c.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [c.c_ulong, c.c_int, c.c_ulong]
    kernel.OpenProcess.restype = c.c_void_p
    kernel.ReadProcessMemory.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t, c.POINTER(c.c_size_t)]
    kernel.CloseHandle.argtypes = [c.c_void_p]
    psapi.EnumProcessModulesEx.argtypes = [c.c_void_p, c.c_void_p, c.c_ulong, c.POINTER(c.c_ulong), c.c_ulong]
    psapi.GetModuleBaseNameW.argtypes = [c.c_void_p, c.c_void_p, c.c_wchar_p, c.c_ulong]
    handle = kernel.OpenProcess(0x410, False, int(pid))
    if not handle:
        raise c.WinError(c.get_last_error())

    def read(address, size):
        if not address or not 0 <= size <= 4_000_000:
            raise ValueError('Invalid or racing pointer/size')
        buffer = c.create_string_buffer(size)
        got = c.c_size_t()
        if not kernel.ReadProcessMemory(handle, address, buffer, size, c.byref(got)) or got.value != size:
            raise c.WinError(c.get_last_error())
        return buffer.raw

    def ptr(address):
        return struct.unpack('<Q', read(address, 8))[0]

    def unicode(address):
        header = read(address, 48)
        length = struct.unpack_from('<q', header, 16)[0]
        flags = struct.unpack_from('<I', header, 32)[0]
        if not 0 <= length <= 8192 or not flags & 32:
            return '<noncompact string>'
        kind = (flags >> 2) & 7
        offset = 48 if flags & 64 else 72
        return read(address + offset, length * kind).decode({1: 'latin1', 2: 'utf-16-le', 4: 'utf-32-le'}[kind])

    try:
        modules = (c.c_void_p * 2048)()
        needed = c.c_ulong()
        if not psapi.EnumProcessModulesEx(handle, modules, c.sizeof(modules), c.byref(needed), 3):
            raise c.WinError(c.get_last_error())
        remote_base = None
        for module in modules[:needed.value // 8]:
            name = c.create_unicode_buffer(512)
            psapi.GetModuleBaseNameW(handle, module, name, len(name))
            if name.value.lower() == 'python311.dll':
                remote_base = module
                break
        if remote_base is None:
            raise RuntimeError('Target has no python311.dll')
        local_base = c.pythonapi._handle
        runtime = c.addressof(c.c_char.in_dll(c.pythonapi, '_PyRuntime'))
        c.pythonapi.PyInterpreterState_Head.restype = c.c_void_p
        c.pythonapi.PyInterpreterState_ThreadHead.argtypes = [c.c_void_p]
        c.pythonapi.PyInterpreterState_ThreadHead.restype = c.c_void_p
        local_interp = c.pythonapi.PyInterpreterState_Head()
        local_thread = c.pythonapi.PyInterpreterState_ThreadHead(local_interp)
        interp_offset = c.string_at(runtime, 1024).index(struct.pack('<Q', local_interp))
        thread_offset = c.string_at(local_interp, 4096).index(struct.pack('<Q', local_thread))
        interp = ptr(remote_base + runtime - local_base + interp_offset)
        thread = ptr(interp + thread_offset)
        seen_threads = set()
        while thread and thread not in seen_threads:
            seen_threads.add(thread)
            cframe = ptr(thread + 56)
            frame = ptr(cframe + 8) if cframe else 0
            print(f'THREAD {thread:#x}')
            seen = set()
            seen_cframes = set()
            while cframe and len(seen) < 160:
                if not frame or frame in seen:
                    if cframe in seen_cframes:
                        break
                    seen_cframes.add(cframe)
                    cframe = ptr(cframe + 16)
                    frame = ptr(cframe + 8) if cframe else 0
                    continue
                seen.add(frame)
                header = read(frame, 72)
                code = struct.unpack_from('<Q', header, 32)[0]
                instruction = struct.unpack_from('<Q', header, 56)[0]
                code_header = read(code, 184)
                filename = unicode(struct.unpack_from('<Q', code_header, 112)[0])
                name = unicode(struct.unpack_from('<Q', code_header, 120)[0])
                first = struct.unpack_from('<i', code_header, 72)[0]
                table = struct.unpack_from('<Q', code_header, 136)[0]
                table_size = struct.unpack('<q', read(table + 16, 8))[0]
                line_table = read(table + 32, table_size)
                offset = instruction - code - 184
                units = struct.unpack_from('<q', code_header, 16)[0]
                if not 0 <= units <= 2_000_000:
                    raise ValueError('Invalid or racing code size')
                proxy = (lambda: None).__code__.replace(co_code=b'\x09\x00' * units,
                    co_linetable=line_table, co_firstlineno=first)
                line = next((line for start, end, line in proxy.co_lines() if start <= offset < end), first)
                print(f'  {filename}:{line} {name}')
                if name == '_class_surface_ssa_program':
                    names = struct.unpack_from('<Q', code_header, 96)[0]
                    count = struct.unpack('<q', read(names + 16, 8))[0]
                    for index in range(count):
                        local_name = unicode(ptr(names + 24 + 8 * index))
                        if local_name not in {'changed', 'next_value_id', 'caller_symbol', 'source_id', 'callee_id', 'frame_round'}:
                            continue
                        value = ptr(frame + 72 + index * 8)
                        if not value:
                            continue
                        type_address = ptr(value + 8)
                        local_cell = c.addressof(c.c_char.in_dll(c.pythonapi, 'PyCell_Type'))
                        if type_address == remote_base + local_cell - local_base:
                            value = ptr(value + 16)
                            if not value:
                                continue
                            type_address = ptr(value + 8)
                        local_long = c.addressof(c.c_char.in_dll(c.pythonapi, 'PyLong_Type'))
                        local_bool = c.addressof(c.c_char.in_dll(c.pythonapi, 'PyBool_Type'))
                        local_str = c.addressof(c.c_char.in_dll(c.pythonapi, 'PyUnicode_Type'))
                        if type_address in {remote_base + local_long - local_base, remote_base + local_bool - local_base}:
                            size = struct.unpack('<q', read(value + 16, 8))[0]
                            if abs(size) <= 4:
                                digits = read(value + 24, max(1, abs(size)) * 4)
                                held = sum(struct.unpack_from('<I', digits, i * 4)[0] << (30 * i) for i in range(abs(size)))
                                print(f'    {local_name}={held if size >= 0 else -held}')
                        elif type_address == remote_base + local_str - local_base:
                            print(f'    {local_name}={unicode(value)}')
                frame = struct.unpack_from('<Q', header, 48)[0]
            thread = ptr(thread + 8)
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    main(int(sys.argv[1]))
