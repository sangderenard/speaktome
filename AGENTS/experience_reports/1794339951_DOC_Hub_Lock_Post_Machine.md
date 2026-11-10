# Hub-lock posts in the existing Machine catalogue

Added `hub-lock-posts` directly to `engine_toy/machines.py` using the existing Machine, MachinePart, MachineLine and PartPort records. Four passive posts have feet, arms, hub plates and latches. The standard engine mesh builds all parts. Each post exposes hub and ground structural-mount ports with the existing bolted-flange joint; matching ports mate without changing their authored positions.

No custom simulator, hidden wheel brake, or validator preload was added. This catalogue machine is not yet installed into the live game car spawn. The host must provide actual hub positions, mate hubs and ground, and feed those connections into the existing mechanical runtime. That integration remains separate from the verified catalogue/mesh implementation.

Verification: 23 tests passed across test_hub_lock_posts.py, test_machine_options.py and test_machine_package.py. One pre-existing cffi imp deprecation warning.

## Prompt History

User: "Check if the machine is built by the validator that might bake in prestress, have the vehicle off when you load, make a little mesh for posts that lock the hubs like, a parking violation lock, no reason other than it's a cute machine to add to the engines database of machines and it gives us something simpler than the validator for a car to spawn in"
