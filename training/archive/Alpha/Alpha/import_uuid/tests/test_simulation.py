# test_simulation.py

import pytest
from simulation import ThreadManager, PhysicalObject, CornerstoneShell
import torch

@pytest.fixture
def thread_manager():
    return ThreadManager()

def test_thread_manager_register_object(thread_manager):
    obj_id = "test_object"
    thread_manager.register_object(obj_id)
    assert obj_id in thread_manager.locks
    assert obj_id in thread_manager.mailboxes
    assert obj_id in thread_manager.tokens
    assert isinstance(thread_manager.locks[obj_id], threading.Lock)
    assert isinstance(thread_manager.mailboxes[obj_id], deque)
    assert isinstance(thread_manager.tokens[obj_id], str)

def test_thread_manager_acquire_and_release_lock(thread_manager):
    obj_id = "test_object"
    thread_manager.register_object(obj_id)
    token = thread_manager.tokens[obj_id]
    
    # Acquire lock
    acquired = thread_manager.acquire_lock(obj_id, token)
    assert acquired == True
    assert thread_manager.locks[obj_id].locked()
    
    # Attempt to acquire lock again should fail
    acquired_again = thread_manager.acquire_lock(obj_id, token)
    assert acquired_again == False
    
    # Release lock
    thread_manager.release_lock(obj_id)
    assert not thread_manager.locks[obj_id].locked()
    
    # New token should be different
    new_token = thread_manager.tokens[obj_id]
    assert new_token != token

def test_thread_manager_send_and_receive_message(thread_manager):
    obj_id = "test_object"
    thread_manager.register_object(obj_id)
    message = "Hello, World!"
    
    # Send message
    thread_manager.send_message(obj_id, message)
    assert len(thread_manager.mailboxes[obj_id]) == 1
    
    # Receive message
    received = thread_manager.receive_message(obj_id)
    assert received == message
    assert len(thread_manager.mailboxes[obj_id]) == 0
    
    # Receive from empty mailbox
    received_none = thread_manager.receive_message(obj_id)
    assert received_none is None

def test_physical_object_initialization(thread_manager):
    obj_id = "test_physical_object"
    position = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    obj = PhysicalObject(obj_id, position, orientation, thread_manager)
    
    assert obj.object_id == obj_id
    assert torch.equal(obj.position, torch.tensor(position, dtype=torch.float64))
    assert torch.equal(obj.orientation, torch.tensor(orientation, dtype=torch.float64))
    assert obj.object_id in thread_manager.locks
    assert obj.object_id in thread_manager.mailboxes
    assert obj.object_id in thread_manager.tokens
    assert 'spherical' in obj.vertex_buffers
    assert 'rectangular' in obj.vertex_buffers
    assert obj.arbitrary_shape_buffer is None

def test_physical_object_update(thread_manager, caplog):
    obj_id = "test_physical_object"
    position = [1.0, 1.0, 1.0]
    orientation = [0.0, 0.0, 0.0]
    obj = PhysicalObject(obj_id, position, orientation, thread_manager)
    
    with caplog.at_level(logging.DEBUG):
        obj.update(dt=1.0)
    
    # Check if update was performed
    # Since perform_update is a placeholder, only logs are checked
    assert f"PhysicalObject '{obj_id}' performing update with dt=1.0." in caplog.text

def test_cornerstone_shell_initialization(thread_manager):
    obj_id = "cornerstone_test"
    position = [1.0, 1.0, 1.0]
    orientation = [0.0, 0.0, 0.0]
    tolerances = {'+X': 0.02, '-X': 0.02, '+Y': 0.02, '-Y': 0.02, '+Z': 0.02, '-Z': 0.02}
    shell = CornerstoneShell(obj_id, position, orientation, tolerances, thread_manager)
    
    assert shell.object_id == obj_id
    assert shell.machining_tolerances == tolerances
    assert shell.arbitrary_shape_buffer is not None
    assert shell.arbitrary_shape_buffer.shape == (8, 3)

def test_cornerstone_shell_imperfect_cube(thread_manager):
    obj_id = "cornerstone_test"
    position = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    tolerances = {'+X': 0.02, '-X': 0.02, '+Y': 0.02, '-Y': 0.02, '+Z': 0.02, '-Z': 0.02}
    shell = CornerstoneShell(obj_id, position, orientation, tolerances, thread_manager)
    
    # Verify that each vertex is within the specified tolerance
    base_vertices = shell.generate_unit_cube_vertices()
    for idx, vertex in enumerate(shell.arbitrary_shape_buffer):
        base_vertex = base_vertices[idx]
        diff = torch.abs(vertex - base_vertex)
        max_diff = diff.max().item()
        face = shell.determine_face(vertex)
        tolerance = tolerances.get(face, 0.01)
        assert max_diff <= tolerance, f"Vertex {idx} exceeds tolerance: {max_diff} > {tolerance}"

def test_cornerstone_shell_update(thread_manager, caplog):
    obj_id = "cornerstone_test"
    position = [1.0, 1.0, 1.0]
    orientation = [0.0, 0.0, 0.0]
    tolerances = {'+X': 0.02, '-X': 0.02, '+Y': 0.02, '-Y': 0.02, '+Z': 0.02, '-Z': 0.02}
    shell = CornerstoneShell(obj_id, position, orientation, tolerances, thread_manager)
    
    with caplog.at_level(logging.DEBUG):
        shell.update(dt=2.0)
    
    # Check if update was performed
    assert f"CornerstoneShell '{obj_id}' expanded to position" in caplog.text
    assert f"PhysicalObject '{obj_id}' performing update with dt=2.0." in caplog.text
