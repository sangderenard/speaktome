const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const cgpt_path = "tensors/tensors/models/c_gpt";
    const cgpt = b.addSharedLibrary("cgpt", .{ .root_source_file = .{ .path = cgpt_path ++ "/cgpt.c" } });
    cgpt.setTarget(target);
    cgpt.setBuildMode(mode);
    cgpt.linkLibC();
    b.installArtifact(cgpt);


    const libs = &[_]struct{ name: []const u8, url: []const u8 }{
        .{ .name = "blis", .url = "https://github.com/flame/blis.git" },
        .{ .name = "blasfeo", .url = "https://github.com/giaf/blasfeo.git" },
        .{ .name = "sleef", .url = "https://github.com/shibatch/sleef.git" },
        .{ .name = "cephes", .url = "https://github.com/jeremybarnes/cephes.git" },
        .{ .name = "cnpy", .url = "https://github.com/rogersce/cnpy.git" },
        .{ .name = "klib", .url = "https://github.com/attractivechaos/klib.git" },
        .{ .name = "uthash", .url = "https://github.com/troydhanson/uthash.git" },
        .{ .name = "marisa-trie", .url = "https://github.com/s-yata/marisa-trie.git" },
        .{ .name = "jsmn", .url = "https://github.com/zserge/jsmn.git" },
        .{ .name = "cJSON", .url = "https://github.com/DaveGamble/cJSON.git" },
        .{ .name = "pcg-c", .url = "https://github.com/imneme/pcg-c.git" },
        .{ .name = "xoshiro-c", .url = "https://github.com/brycelelbach/xoshiro-c.git" },
        .{ .name = "stb", .url = "https://github.com/nothings/stb.git" },
        .{ .name = "docopt.c", .url = "https://github.com/docopt/docopt.c.git" },
        .{ .name = "linenoise", .url = "https://github.com/antirez/linenoise.git" },
    };

    for (libs) |lib| {
        const dir_path = "third_party/" ++ lib.name;
        const dir = std.fs.cwd().openDir(dir_path, .{}) catch null;
        if (dir == null) {
            const res = b.exec(&[_][]const u8{
                "git", "clone", "--depth", "1", lib.url, dir_path,
            }) catch std.log.err("failed to clone {s}", .{lib.name});
            _ = res;
        }
    }
}
