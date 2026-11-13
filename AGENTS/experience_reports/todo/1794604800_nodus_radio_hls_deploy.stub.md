# Radio HLS deployment and browser proof

- [ ] Install a Nodus `nodus_radio.exe` build containing PR #73 in the operator-owned product location; retain its build receipt and DLL dependencies.
- [ ] Configure an authorized file-source jukebox and `jukebox.native.hls` in the instance's private `StatePaths.radio_json`, with a chosen catalogue and actual protected core roots. Keep source rights and account access explicit.
- [ ] Start the jukebox through its existing admin control and verify the playlist and numbered segments over Apache HTTPS with a `view_site` principal. Verify a denied principal cannot read them.
- [ ] Play the stream in a native-HLS browser and measure live segment delay and switch behavior across two songs; then decide whether to vendor a client HLS player for browsers without native support.
- [ ] If FFmpeg is vendored, pin the binary, license receipt and AAC/HLS feature set before replacing the installed executable path.
