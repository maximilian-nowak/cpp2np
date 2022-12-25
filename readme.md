# cpp2np

## Requirements
    Python >= 3.9
    numpy >= 1.23.4
    g++ >= 9.3.0
    GNU Make >= 4.2.1

## Installation

Zur Installation führt man den 'make' Befehl im Root-Verzeichnis des Projektes aus.

    user@HOST:~/cpp2np$ make

Das cpp2np Modul wird automatisch gebaut und im environment installiert.
    
## Problem mit setup.py

Falls die Installation nicht klappt, könnte eine Lösung sein, die .so Datei manuell zum Pfad
der dynamic linked libraries hinzuzufügen.

Dazu führt man folgenden Befehl im Root-Verzeichnis des Projektes aus:

    user@HOST:~/cpp2np$ echo export LD_LIBRARY_PATH=$(pwd)/build/lib.linux-x86_64-cpython-310 >> ~/.bashrc

## Demo

### import module

```python
import cpp2np as c2n
import numpy as np
```

### Get pointer to 2x2 std::array allocated by c++:

```python
>>> pointer, shape = c2n.array_2x2()
>>> print(pointer)
>>> print(shape)

24245840
(2, 2)
```

### wrap pointer in numpy array

```python
>>> wrapper = c2n.wrap(pointer, shape, dtype=np.dtype("int32"))
>>> print(wrapper)
>>> print(type(wrapper))

[[1 2]
[3 4]]

<class 'numpy.ndarray'>
```

### change value in numpy array

```python
>>> wrapper[0,0] = 255
>>> print(wrapper)

[[255   2]
[  3   4]]
```

### delete numpy array and create new wrapper from same pointer

```python
>>> del wrapper
>>> wrapper2 = c2n.wrap(pointer, shape, dtype=np.dtype("int32"))
>>> print(wrapper2)

[[255   2]
[  3   4]]
```

(We observe the change of value in first wrapper was done on the original memory buffer,
as it also shows up in the new wrapper. Also deleting the wrapper did not delete the buffer.)


### Get information of underlying data of the wrapper

```python
>>> print(c2n.descr(wrapper2))

{'data': 24245840, 'ndim': 2, 'shape': (2, 2), 'typestr': '<i4'}
```

### To check if data is contiguous we can look into flags attribute of the numpy array

```python
>>> print("C contiguous: " + str(wrapper2.flags['C_CONTIGUOUS']))
>>> print("F contiguous: " + str(wrapper2.flags['F_CONTIGUOUS']))

C contiguous: True
F contiguous: False
```    
    
### Flags overview

```python
>>> wrapper2.flags

C_CONTIGUOUS : True
F_CONTIGUOUS : False
OWNDATA : False
WRITEABLE : True
ALIGNED : True
WRITEBACKIFCOPY : False
```

### Free the memory of the c++ array explicitly

```python
>>> c2n.free(pointer)
>>> print(wrapper2)
```
    [[24407120        0]
    [19943440        0]]


We observe that the numpy array is pointing nowhere as the original buffer was freed on the c++ side.