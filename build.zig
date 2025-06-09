const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const cgpt_path = "speaktome/tensors/models/c_gpt";
    const cgpt = b.addSharedLibrary("cgpt", .{ .root_source_file = .{ .path = cgpt_path ++ "/cgpt.c" } });
    cgpt.setTarget(target);
    cgpt.setBuildMode(mode);
    cgpt.linkLibC();
    b.installArtifact(cgpt);

    // Simple check for vendored BLIS; attempt fetch if missing.
    const blis_dir = std.fs.cwd().openDir("third_party/blis", .{}) catch null;
    if (blis_dir == null) {
        const res = b.exec(&[_][]const u8{
            "git", "clone", "--depth", "1",
            "https://github.com/flame/blis.git", "third_party/blis",
        }) catch std.log.err("failed to clone blis", .{});
        _ = res;
    }
}
