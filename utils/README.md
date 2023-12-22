# Utility Functions

Utility functions and classes that are shared across different parts of the project.
This collection of utilities is designed to reduce code duplication and promote reusable code patterns.

#### Structure

The `utils` submodule has the following structure:

```
utils
├── __init__.py
├── README.md
├── src
│   ├── __init__.py
│   └── utils.py
└── tests
```

#### Importing Utilities

Thanks to the way the __init__.py files are structured, you can import any utility function directly from the utils
submodule.

```python
from utils import your_utility_function
```

#### Available Utilities

The `utils.py` file contains various functions and classes that can be used throughout the project. Refer to the
comments and documentation within the code for detailed descriptions of each utility's purpose and usage.

#### Tests

The `tests` directory can be used to store unit tests for the utility functions and classes. Regular testing ensures
that these essential components remain robust and error-free, especially when updates or modifications are made
elsewhere in the codebase.
