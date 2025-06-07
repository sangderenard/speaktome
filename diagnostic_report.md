
## PyTest Session Header 

  Test Session Timestamp: 20250607_093020

## Faculty Information 

  Environment variable SPEAKTOME_FACULTY is NOT set.
  Using auto-detected faculty for tests: PYGEO.

## Module Headers 

  ===== speaktome\__init__.py =====
  """SpeakToMe beam search package."""
  
  from .faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV
  # --- END HEADER ---
  ===== speaktome\config.py =====
  # Third-party imports
  import os
  
  from .faculty import Faculty, DEFAULT_FACULTY
  # --- END HEADER ---
  ===== speaktome\core\__init__.py =====
  # --- END HEADER ---
  ===== speaktome\core\beam_graph_operator.py =====
  # Standard library imports
  import json
  from typing import List, Set, Optional, Tuple, Dict, Callable, TYPE_CHECKING, Union
  
  # Third-party imports
  import torch
  
  # Local application/library specific imports
  from .beam_tree_node import BeamTreeNode # Assuming BeamTreeNode is in beam_tree_node.py
  # --- END HEADER ---
  ===== speaktome\core\beam_retirement_manager.py =====
  # Standard library imports
  import queue as py_queue
  import threading
  from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
  
  # Third-party imports
  import torch
  # --- END HEADER ---
  ===== speaktome\core\beam_search.py =====
  # Standard library imports
  from __future__ import annotations
  from typing import List, Optional, Tuple, Callable, Dict, Any, TYPE_CHECKING
  import math
  
  # Third-party imports
  if TYPE_CHECKING:  # pragma: no cover - type hints only
      import torch
  
  from .. import Faculty
  
  FACULTY_REQUIREMENT = Faculty.TORCH
  
  # Local application/library specific imports
  # Please adjust these import paths based on your actual project structure.
  from .. import config
  from ..config import GPU_LIMIT, LENGTH_LIMIT
  from .beam_graph_operator import BeamGraphOperator
  from .beam_search_instruction import BeamSearchInstruction
  from .scorer import Scorer
  from .beam_retirement_manager import BeamRetirementManager
  from .compressed_beam_tree import CompressedBeamTree
  from .tensor_abstraction import (
      AbstractTensorOperations,
  )
  from .model_abstraction import (
      AbstractModelWrapper,
      PyTorchModelWrapper,
  )
  from .lookahead_controller import LookaheadConfig, LookaheadController
  # --- END HEADER ---
  ===== speaktome\core\beam_search_instruction.py =====
  # Standard library imports
  from typing import Any, Callable, Dict, List, Optional, Tuple
  import torch
  # Local application/library specific imports
  from .scorer import Scorer # Assuming Scorer is in scorer.py
  # --- END HEADER ---
  ===== speaktome\core\beam_tree_node.py =====
  # Standard library imports
  from typing import List, Optional
  
  # Third-party imports
  import torch
  # --- END HEADER ---
  ===== speaktome\core\beam_tree_visualizer.py =====
  # Standard library imports
  import collections
  from typing import Optional, List, TYPE_CHECKING
  
  # Third-party imports
  import torch
  
  if TYPE_CHECKING:
      from transformers import PreTrainedTokenizer
  
  from ..util.lazy_loader import lazy_install
  # --- END HEADER ---
  ===== speaktome\core\compressed_beam_tree.py =====
  # Standard library imports
  from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
  
  # Third-party imports
  import torch
  from ..util.lazy_loader import lazy_install
  
  if TYPE_CHECKING:
      from transformers import PreTrainedTokenizer  # For type hinting
  
  if TYPE_CHECKING:
      from torch_geometric.data import Data as PyGData  # pragma: no cover
  
  # Local application/library specific imports
  from .beam_tree_node import BeamTreeNode # Assuming BeamTreeNode is in beam_tree_node.py
  # --- END HEADER ---
  ===== speaktome\core\human_pilot_controller.py =====
  # Standard library imports
  from typing import List, Optional, TYPE_CHECKING
  
  # Third-party imports
  import torch # type: ignore[import-untyped]
  from torch_geometric.data import Data as PyGData # type: ignore[import-untyped] # Moved for runtime availability
  
  # Local application/library specific imports
  from .beam_search import BeamSearch
  from .beam_search_instruction import BeamSearchInstruction
  # --- END HEADER ---
  ===== speaktome\core\human_scorer_policy_manager.py =====
  # Standard library imports
  import json
  from typing import Dict, Callable, Optional, List, Tuple
  
  # Local application/library specific imports
  from .scorer import Scorer
  # --- END HEADER ---
  ===== speaktome\core\lookahead_controller.py =====
  # Standard library imports
  from __future__ import annotations
  from typing import List, Tuple, Callable, Any, Set
  
  # Local application/library specific imports
  from .beam_search_instruction import BeamSearchInstruction # Assuming this is in its own file
  from .tensor_abstraction import AbstractTensorOperations
  from .model_abstraction import AbstractModelWrapper
  # --- END HEADER ---
  ===== speaktome\core\model_abstraction.py =====
  from __future__ import annotations
  
  from abc import ABC, abstractmethod
  from typing import Any, Dict
  # --- END HEADER ---
  ===== speaktome\core\scorer.py =====
  """Scoring utilities for beam search and research experimentation.
  
  scoring functions.  The design favours vectorised tensor operations so models
  and heuristics can scale gracefully.  Future versions will introduce a queue
  and mailbox mechanism so that tokenisation, model inference, and scoring can be
  scheduled in worker threads.  This will allow the scoring pipeline to operate
  on arbitrary array types while delivering results to dynamic mailboxes for
  maximum throughput.
  """
  
  from typing import Callable, Dict
  
  import os
  import torch
  import torch.nn.functional as F
  import queue
  
  from .tensor_abstraction import (
      AbstractTensorOperations,
      get_tensor_operations,
  )
  
  from ..util.lazy_loader import lazy_import, optional_import
  from .. import config
  # --- END HEADER ---
  ===== speaktome\core\tensor_abstraction.py =====
  from abc import ABC, abstractmethod
  from typing import Any, Tuple, Optional, List, Union
  import math
  
  from ..faculty import Faculty, DEFAULT_FACULTY
  from .. import config
  # --- END HEADER ---
  ===== speaktome\cpu_demo.py =====
  """CPU-only demo exercising :class:`LookaheadController`.
  
  This lightweight path demonstrates how the project can operate with
  either NumPy or a pure Python fallback. A simple random model drives the
  lookahead search using the generic tensor and model wrappers. The demo
  prints the top ``k`` results after ``d`` lookahead steps.
  """
  
  import argparse
  from typing import Any, Dict
  
  from . import Faculty
  
  FACULTY_REQUIREMENT = Faculty.PURE_PYTHON
  try:
      import numpy as np
      NUMPY_AVAILABLE = True
  except ModuleNotFoundError:  # pragma: no cover - optional dependency
      NUMPY_AVAILABLE = False
      np = None  # type: ignore
  
  from .util.token_vocab import TokenVocabulary
  from .core.tensor_abstraction import get_tensor_operations
  from .core.model_abstraction import AbstractModelWrapper
  from .core.lookahead_controller import LookaheadController, LookaheadConfig
  # --- END HEADER ---
  ===== speaktome\domains\__init__.py =====
  # --- END HEADER ---
  ===== speaktome\domains\geo\__init__.py =====
  # --- END HEADER ---
  ===== speaktome\domains\geo\pyg_graph_controller.py =====
  # Standard library imports
  from typing import Optional, List, TYPE_CHECKING
  
  # Third-party imports
  import torch
  
  from ...util.lazy_loader import lazy_import
  if TYPE_CHECKING:
      from torch_geometric.data import Data as PyGData
  
  # Local application/library specific imports
  from ...core.beam_search import BeamSearch
  from .pygeo_mind import PyGeoMind
  # Local application/library specific imports
  from ...core.human_scorer_policy_manager import HumanScorerPolicyManager
  from ...core.beam_search_instruction import BeamSearchInstruction
  # Ensure this import points to the new location of BeamTreeVisualizer
  # Import both visualizers from beam_tree_visualizer.py
  from ...core.beam_tree_visualizer import BeamTreeVisualizer
  from ...config import get_sentence_transformer_model
  # --- END HEADER ---
  ===== speaktome\domains\geo\pygeo_mind.py =====
  # Standard library imports
  import collections
  from typing import Dict, List, Tuple, TYPE_CHECKING
  
  # Third-party imports
  import torch
  
  from ...faculty import Faculty
  
  FACULTY_REQUIREMENT = Faculty.PYGEO
  
  from ...util.lazy_loader import lazy_install
  if TYPE_CHECKING:
      from torch_geometric.data import Data as PyGData
      import torch_geometric.nn as pyg_nn
  
  # Local application/library specific imports
  from ...core.beam_search_instruction import BeamSearchInstruction
  from ...core.compressed_beam_tree import CompressedBeamTree
  from ...core.human_scorer_policy_manager import HumanScorerPolicyManager
  from ...core.scorer import Scorer
  # --- END HEADER ---
  ===== speaktome\faculty.py =====
  from __future__ import annotations
  
  """Faculty levels for runtime resources."""
  
  import os
  from enum import Enum, auto
  # --- END HEADER ---
  ===== speaktome\speaktome.py =====
  # Standard library imports
  from typing import List, Tuple, Dict, Optional, Any, Callable, TYPE_CHECKING
  import argparse
  
  # Third-party imports
  from . import Faculty, DEFAULT_FACULTY
  try:
      import torch
  except ModuleNotFoundError:  # pragma: no cover - runtime message only
      torch = None
  
  TORCH_AVAILABLE = DEFAULT_FACULTY in (Faculty.TORCH, Faculty.PYGEO)
  if not TORCH_AVAILABLE:
      print("PyTorch is not installed. Running CPU-only demo mode.")
  try:
      from transformers import PreTrainedTokenizer
      TRANSFORMERS_AVAILABLE = True
  except ModuleNotFoundError:  # pragma: no cover - runtime message only
      PreTrainedTokenizer = None  # type: ignore
      TRANSFORMERS_AVAILABLE = False
      print("Transformers is not installed. Running CPU-only demo mode.")
  
  from .util.lazy_loader import lazy_import
  if TYPE_CHECKING:
      from torch_geometric.data import Data as PyGData
      import torch_geometric.nn as pyg_nn
      from sentence_transformers import SentenceTransformer
  
  # Local application/library specific imports
  # Please adjust these import paths if your project structure differs.
  # Import configuration dynamically to allow device changes at runtime
  from . import config
  from .config import get_sentence_transformer_model, GPU_LIMIT, LENGTH_LIMIT
  # --- END HEADER ---
  ===== speaktome\util\__init__.py =====
  # --- END HEADER ---
  ===== speaktome\util\array_utils.py =====
  """Utility helpers for array operations with optional torch support."""
  
  import numpy as np
  # --- END HEADER ---
  ===== speaktome\util\cli_permutations.py =====
  from __future__ import annotations
  
  from itertools import product
  from typing import Any, Dict, Iterable, List, Sequence, Tuple
  # --- END HEADER ---
  ===== speaktome\util\lazy_loader.py =====
  import importlib
  import subprocess
  import sys
  from functools import lru_cache
  # --- END HEADER ---
  ===== speaktome\util\token_vocab.py =====
  # --- END HEADER ---
  ===== AGENTS\tools\fetch_official_time.py =====
  #!/usr/bin/env python3
  """Fetch official UTC time from a trusted source and optionally set system time.
  
  This script contacts ``worldtimeapi.org`` to retrieve the current UTC timestamp.
  The timestamp can be written to a file or printed. With ``--apply`` the script
  attempts to set the local system clock using ``sudo date`` on POSIX systems or
  ``Set-Date`` on Windows. Use with caution and appropriate privileges.
  """
  
  from __future__ import annotations
  
  import argparse
  import json
  import platform
  import subprocess
  import sys
  import urllib.request
  # --- END HEADER ---
  ===== AGENTS\tools\header_guard_precommit.py =====
  #!/usr/bin/env python3
  """
  Pre-commit hook: Enforces HEADER, ``@staticmethod test()`` and ``# --- END HEADER ---`` placement.
  Blocks commit if any staged ``.py`` file is missing these requirements.
  Set environment variable ``SKIP_HEADER_GUARD`` to disable.
  
  Prototype author: GitHub Copilot (o3[4.1 sic]), for the SPEAKTOME agent ecosystem.
  License: MIT
  """
  
  import sys
  import subprocess
  import ast
  import os
  from pathlib import Path
  # --- END HEADER ---
  ===== AGENTS\tools\pretty_logger.py =====
  """Pretty logging with contextual headers and markdown formatting."""
  
  import logging
  import sys
  from contextlib import contextmanager
  from dataclasses import dataclass
  from typing import List, Optional
  
  # --- END HEADER ---
============================= test session starts =============================
platform win32 -- Python 3.10.6, pytest-8.4.0, pluggy-1.6.0
rootdir: C:\Apache24\htdocs\AI\speaktome_project
configfile: pyproject.toml
collected 37 items

tests\test_all_classes.py ..FFFFF...FF.FF...                             [ 48%]
tests\test_cli.py .F                                                     [ 54%]
tests\test_cli_permutations.py .                                         [ 56%]
tests\test_faculty.py .FFF........                                       [ 89%]
tests\test_pure_python_tensor_ops.py ...                                 [ 97%]
tests\test_validate_guestbook.py .                                       [100%]

================================== FAILURES ===================================
_ test_class_instantiation[speaktome.core.beam_graph_operator-BeamGraphOperator] _

mod_name = 'speaktome.core.beam_graph_operator', cls_name = 'BeamGraphOperator'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: BeamGraphOperator.__init__() missing 1 required positional argument: 'tree'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.core.beam_graph_operator', cls_name = 'BeamGraphOperator'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate BeamGraphOperator: BeamGraphOperator.__init__() missing 1 required positional argument: 'tree'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_graph_operator-BeamGraphOperator] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_graph_operator-BeamGraphOperator]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for BeamGraphOperator
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating BeamGraphOperator
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate BeamGraphOperator: BeamGraphOperator.__init__() missing 1 required positional argument: 'tree'
_ test_class_instantiation[speaktome.core.beam_retirement_manager-BeamRetirementManager] _

mod_name = 'speaktome.core.beam_retirement_manager'
cls_name = 'BeamRetirementManager'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: BeamRetirementManager.__init__() missing 1 required positional argument: 'tree'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.core.beam_retirement_manager'
cls_name = 'BeamRetirementManager'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate BeamRetirementManager: BeamRetirementManager.__init__() missing 1 required positional argument: 'tree'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_retirement_manager-BeamRetirementManager] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_retirement_manager-BeamRetirementManager]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for BeamRetirementManager
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating BeamRetirementManager
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate BeamRetirementManager: BeamRetirementManager.__init__() missing 1 required positional argument: 'tree'
_______ test_class_instantiation[speaktome.core.beam_search-BeamSearch] _______

mod_name = 'speaktome.core.beam_search', cls_name = 'BeamSearch'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: BeamSearch.__init__() missing 1 required positional argument: 'scorer'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.core.beam_search', cls_name = 'BeamSearch'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate BeamSearch: BeamSearch.__init__() missing 1 required positional argument: 'scorer'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_search-BeamSearch] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_search-BeamSearch]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for BeamSearch
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating BeamSearch
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate BeamSearch: BeamSearch.__init__() missing 1 required positional argument: 'scorer'
_ test_class_instantiation[speaktome.core.beam_search_instruction-BeamSearchInstruction] _

mod_name = 'speaktome.core.beam_search_instruction'
cls_name = 'BeamSearchInstruction'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: BeamSearchInstruction.__init__() missing 2 required positional arguments: 'node_id' and 'action'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.core.beam_search_instruction'
cls_name = 'BeamSearchInstruction'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate BeamSearchInstruction: BeamSearchInstruction.__init__() missing 2 required positional arguments: 'node_id' and 'action'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_search_instruction-BeamSearchInstruction] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_search_instruction-BeamSearchInstruction]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for BeamSearchInstruction
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating BeamSearchInstruction
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate BeamSearchInstruction: BeamSearchInstruction.__init__() missing 2 required positional arguments: 'node_id' and 'action'
____ test_class_instantiation[speaktome.core.beam_tree_node-BeamTreeNode] _____

mod_name = 'speaktome.core.beam_tree_node', cls_name = 'BeamTreeNode'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: BeamTreeNode.__init__() missing 4 required positional arguments: 'token', 'score', 'parent_node_idx', and 'depth'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.core.beam_tree_node', cls_name = 'BeamTreeNode'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate BeamTreeNode: BeamTreeNode.__init__() missing 4 required positional arguments: 'token', 'score', 'parent_node_idx', and 'depth'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_tree_node-BeamTreeNode] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_tree_node-BeamTreeNode]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for BeamTreeNode
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating BeamTreeNode
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate BeamTreeNode: BeamTreeNode.__init__() missing 4 required positional arguments: 'token', 'score', 'parent_node_idx', and 'depth'
_ test_class_instantiation[speaktome.core.human_pilot_controller-HumanPilotController] _

mod_name = 'speaktome.core.human_pilot_controller'
cls_name = 'HumanPilotController'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: HumanPilotController.__init__() missing 1 required positional argument: 'beam_search'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.core.human_pilot_controller'
cls_name = 'HumanPilotController'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate HumanPilotController: HumanPilotController.__init__() missing 1 required positional argument: 'beam_search'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.core.human_pilot_controller-HumanPilotController] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.core.human_pilot_controller-HumanPilotController]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for HumanPilotController
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating HumanPilotController
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate HumanPilotController: HumanPilotController.__init__() missing 1 required positional argument: 'beam_search'
_ test_class_instantiation[speaktome.core.human_scorer_policy_manager-HumanScorerPolicyManager] _

mod_name = 'speaktome.core.human_scorer_policy_manager'
cls_name = 'HumanScorerPolicyManager'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: HumanScorerPolicyManager.__init__() missing 1 required positional argument: 'scorer'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.core.human_scorer_policy_manager'
cls_name = 'HumanScorerPolicyManager'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate HumanScorerPolicyManager: HumanScorerPolicyManager.__init__() missing 1 required positional argument: 'scorer'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.core.human_scorer_policy_manager-HumanScorerPolicyManager] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.core.human_scorer_policy_manager-HumanScorerPolicyManager]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for HumanScorerPolicyManager
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating HumanScorerPolicyManager
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate HumanScorerPolicyManager: HumanScorerPolicyManager.__init__() missing 1 required positional argument: 'scorer'
_ test_class_instantiation[speaktome.domains.geo.pyg_graph_controller-PyGGraphController] _

mod_name = 'speaktome.domains.geo.pyg_graph_controller'
cls_name = 'PyGGraphController'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: PyGGraphController.__init__() missing 1 required positional argument: 'beam_search'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.domains.geo.pyg_graph_controller'
cls_name = 'PyGGraphController'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate PyGGraphController: PyGGraphController.__init__() missing 1 required positional argument: 'beam_search'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.domains.geo.pyg_graph_controller-PyGGraphController] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.domains.geo.pyg_graph_controller-PyGGraphController]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for PyGGraphController
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating PyGGraphController
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate PyGGraphController: PyGGraphController.__init__() missing 1 required positional argument: 'beam_search'
____ test_class_instantiation[speaktome.domains.geo.pygeo_mind-PyGeoMind] _____

mod_name = 'speaktome.domains.geo.pygeo_mind', cls_name = 'PyGeoMind'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
>               cls()
E               TypeError: PyGeoMind.__init__() missing 1 required positional argument: 'scorer'

tests\test_all_classes.py:86: TypeError

During handling of the above exception, another exception occurred:

mod_name = 'speaktome.domains.geo.pygeo_mind', cls_name = 'PyGeoMind'

    @pytest.mark.parametrize("mod_name,cls_name", STUB_MODULES)
    def test_class_instantiation(mod_name: str, cls_name: str) -> None:
        """Import each listed module and instantiate the referenced class."""
        if DEFAULT_FACULTY is Faculty.PURE_PYTHON:
            pytest.skip("Optional libraries not installed; skipping class instantiation tests")
    
        logger.info(f"test_class_instantiation start for {cls_name}")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            logger.error('import failed for %s: %s', mod_name, exc)
            pytest.fail(f"import failed for {mod_name}: {exc}")
    
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.error('class %s not found in %s', cls_name, mod_name)
            pytest.fail(f"class {cls_name} not found in {mod_name}")
    
        logger.info('instantiating %s', cls_name)
        try:
            if cls_name == "PyTorchModelWrapper":
                dummy = types.SimpleNamespace()
                setattr(dummy, 'parameters', lambda: iter([0]))
                cls(dummy)
            else:
                cls()
        except Exception as exc:
            logger.error('failed to instantiate %s: %s', cls_name, exc)
>           pytest.fail(f"failed to instantiate {cls_name}: {exc}")
E           Failed: failed to instantiate PyGeoMind: PyGeoMind.__init__() missing 1 required positional argument: 'scorer'

tests\test_all_classes.py:89: Failed
---------------------------- Captured stdout setup ----------------------------

## tests/test_all_classes.py::test_class_instantiation[speaktome.domains.geo.pygeo_mind-PyGeoMind] 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_all_classes.py::test_class_instantiation[speaktome.domains.geo.pygeo_mind-PyGeoMind]
------------------------------ Captured log call ------------------------------
INFO     tests.test_all_classes:test_all_classes.py:67 test_class_instantiation start for PyGeoMind
INFO     tests.test_all_classes:test_all_classes.py:79 instantiating PyGeoMind
ERROR    tests.test_all_classes:test_all_classes.py:88 failed to instantiate PyGeoMind: PyGeoMind.__init__() missing 1 required positional argument: 'scorer'
___________________________ test_basic_combinations ___________________________

    def test_basic_combinations() -> None:
        """Run a minimal CLI cycle with a few argument permutations."""
        pytest.importorskip('torch', reason='CLI requires torch for full run')
        pytest.importorskip('transformers', reason='CLI requires transformers for full run with torch')
        matrix = CLIArgumentMatrix()
        matrix.add_option('--max_steps', [1])
        matrix.add_option('--safe_mode', [None])
        combos = matrix.generate()
        for combo in combos:
            result = subprocess.run([
                sys.executable,
                '-m', 'speaktome.speaktome',
                *combo,
                'hi'
            ], capture_output=True, text=True)
>           assert result.returncode == 0
E           AssertionError: assert 1 == 0
E            +  where 1 = CompletedProcess(args=['C:\\Apache24\\htdocs\\AI\\speaktome_project\\.venv\\Scripts\\python.exe', '-m', 'speaktome.spe...1, in _ensure_model\n    raise RuntimeError(\nRuntimeError: Transformers is required for the full beam search demo.\n').returncode

tests\test_cli.py:42: AssertionError
---------------------------- Captured stdout setup ----------------------------

## tests/test_cli.py::test_basic_combinations 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_cli.py::test_basic_combinations
_______________________ test_detect_faculty_pure_python _______________________

    def test_detect_faculty_pure_python() -> None:
        """Verify the fallback when no numerical libraries are present."""
        with mock.patch.dict(sys.modules):
            # Ensure these modules are treated as not imported for this test
            sys.modules.pop('torch_geometric', None)
            sys.modules.pop('torch', None)
            sys.modules.pop('numpy', None)
>           assert detect_faculty() == Faculty.PURE_PYTHON

tests\test_faculty.py:30: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
speaktome\faculty.py:36: in detect_faculty
    import torch_geometric  # type: ignore
.venv\lib\site-packages\torch_geometric\__init__.py:3: in <module>
    import torch
.venv\lib\site-packages\torch\__init__.py:2630: in <module>
    class _TritonLibrary:
.venv\lib\site-packages\torch\__init__.py:2631: in _TritonLibrary
    lib = torch.library.Library("triton", "DEF")
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <[AttributeError("'Library' object has no attribute 'kind'") raised in repr()] Library object at 0x1d018345690>
ns = 'triton', kind = 'DEF', dispatch_key = ''

    def __init__(self, ns, kind, dispatch_key=""):
        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError("Unsupported kind: ", kind)
    
        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                ns,
                " is a reserved namespace. Please try creating a library with another name.",
            )
        if torch._running_with_deploy():
            _library.utils.warn_deploy()
            return
    
        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
>       self.m: Optional[Any] = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
E       RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton; please put all of your definitions in a single TORCH_LIBRARY block.  If you were trying to specify implementations, consider using TORCH_LIBRARY_IMPL (which can be duplicated).  If you really intended to define operators for a single namespace in a distributed way, you can use TORCH_LIBRARY_FRAGMENT to explicitly indicate this.  Previous registration of TORCH_LIBRARY was registered at /dev/null:2630; latest registration was registered at /dev/null:2630

.venv\lib\site-packages\torch\library.py:109: RuntimeError
---------------------------- Captured stdout setup ----------------------------

## tests/test_faculty.py::test_detect_faculty_pure_python 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_faculty.py::test_detect_faculty_pure_python
__________________________ test_detect_faculty_numpy __________________________

    def test_detect_faculty_numpy() -> None:
        """Confirm that NumPy alone elevates the faculty to :attr:`Faculty.NUMPY`."""
        with mock.patch.dict(sys.modules):
            sys.modules.pop('torch_geometric', None)
            sys.modules.pop('torch', None)
            sys.modules['numpy'] = mock.MagicMock()  # Simulate numpy is importable
>           assert detect_faculty() == Faculty.NUMPY

tests\test_faculty.py:39: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
speaktome\faculty.py:36: in detect_faculty
    import torch_geometric  # type: ignore
.venv\lib\site-packages\torch_geometric\__init__.py:3: in <module>
    import torch
.venv\lib\site-packages\torch\__init__.py:2630: in <module>
    class _TritonLibrary:
.venv\lib\site-packages\torch\__init__.py:2631: in _TritonLibrary
    lib = torch.library.Library("triton", "DEF")
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <[AttributeError("'Library' object has no attribute 'kind'") raised in repr()] Library object at 0x1d018707b50>
ns = 'triton', kind = 'DEF', dispatch_key = ''

    def __init__(self, ns, kind, dispatch_key=""):
        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError("Unsupported kind: ", kind)
    
        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                ns,
                " is a reserved namespace. Please try creating a library with another name.",
            )
        if torch._running_with_deploy():
            _library.utils.warn_deploy()
            return
    
        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
>       self.m: Optional[Any] = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
E       RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton; please put all of your definitions in a single TORCH_LIBRARY block.  If you were trying to specify implementations, consider using TORCH_LIBRARY_IMPL (which can be duplicated).  If you really intended to define operators for a single namespace in a distributed way, you can use TORCH_LIBRARY_FRAGMENT to explicitly indicate this.  Previous registration of TORCH_LIBRARY was registered at /dev/null:2630; latest registration was registered at /dev/null:2630

.venv\lib\site-packages\torch\library.py:109: RuntimeError
---------------------------- Captured stdout setup ----------------------------

## tests/test_faculty.py::test_detect_faculty_numpy 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_faculty.py::test_detect_faculty_numpy
__________________________ test_detect_faculty_torch __________________________

    def test_detect_faculty_torch() -> None:
        """Check that PyTorch takes precedence when PyG is missing."""
        with mock.patch.dict(sys.modules):
            sys.modules.pop('torch_geometric', None)
            sys.modules['torch'] = mock.MagicMock()  # Simulate torch is importable
            # Numpy might also be present, torch should take precedence
            sys.modules['numpy'] = mock.MagicMock()
>           assert detect_faculty() == Faculty.TORCH

tests\test_faculty.py:49: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
speaktome\faculty.py:36: in detect_faculty
    import torch_geometric  # type: ignore
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    from collections import defaultdict
    
    import torch
    import torch_geometric.typing
    
    from ._compile import compile, is_compiling
    from ._onnx import is_in_onnx_export
    from .index import Index
    from .edge_index import EdgeIndex
    from .seed import seed_everything
    from .home import get_home_dir, set_home_dir
    from .device import is_mps_available, is_xpu_available, device
    from .isinstance import is_torch_instance
    from .debug import is_debug_enabled, debug, set_debug
    
    import torch_geometric.utils
    import torch_geometric.data
    import torch_geometric.sampler
    import torch_geometric.loader
    import torch_geometric.transforms
    import torch_geometric.datasets
    import torch_geometric.nn
    import torch_geometric.explain
    import torch_geometric.profile
    
    from .experimental import (is_experimental_mode_enabled, experimental_mode,
                               set_experimental_mode)
    from .lazy_loader import LazyLoader
    
    contrib = LazyLoader('contrib', globals(), 'torch_geometric.contrib')
    graphgym = LazyLoader('graphgym', globals(), 'torch_geometric.graphgym')
    
    __version__ = '2.6.1'
    
    __all__ = [
        'Index',
        'EdgeIndex',
        'seed_everything',
        'get_home_dir',
        'set_home_dir',
        'compile',
        'is_compiling',
        'is_in_onnx_export',
        'is_mps_available',
        'is_xpu_available',
        'device',
        'is_torch_instance',
        'is_debug_enabled',
        'debug',
        'set_debug',
        'is_experimental_mode_enabled',
        'experimental_mode',
        'set_experimental_mode',
        'torch_geometric',
        '__version__',
    ]
    
    # Serialization ###############################################################
    
>   if torch_geometric.typing.WITH_PT24:
E   AttributeError: partially initialized module 'torch_geometric' has no attribute 'typing' (most likely due to a circular import)

.venv\lib\site-packages\torch_geometric\__init__.py:60: AttributeError
---------------------------- Captured stdout setup ----------------------------

## tests/test_faculty.py::test_detect_faculty_torch 

----------------------------- Captured log setup ------------------------------
INFO     pytest-pretty:pretty_logger.py:51 
## tests/test_faculty.py::test_detect_faculty_torch
=========================== short test summary info ===========================
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_graph_operator-BeamGraphOperator]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_retirement_manager-BeamRetirementManager]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_search-BeamSearch]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_search_instruction-BeamSearchInstruction]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.core.beam_tree_node-BeamTreeNode]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.core.human_pilot_controller-HumanPilotController]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.core.human_scorer_policy_manager-HumanScorerPolicyManager]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.domains.geo.pyg_graph_controller-PyGGraphController]
FAILED tests/test_all_classes.py::test_class_instantiation[speaktome.domains.geo.pygeo_mind-PyGeoMind]
FAILED tests/test_cli.py::test_basic_combinations - AssertionError: assert 1 ...
FAILED tests/test_faculty.py::test_detect_faculty_pure_python - RuntimeError:...
FAILED tests/test_faculty.py::test_detect_faculty_numpy - RuntimeError: Only ...
FAILED tests/test_faculty.py::test_detect_faculty_torch - AttributeError: par...
======================= 13 failed, 24 passed in 27.00s ========================
