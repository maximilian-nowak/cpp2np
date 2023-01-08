# cpp2np

## Requirements

    Python >= 3.9
    numpy >= 1.23.4
    g++ >= 9.3.0
    GNU Make >= 4.2.1

Zusätzliche requirements für `demo2.py`:

    pillow >= 9.4.0
    matplotlib >= 3.6.2
    scipy >= 1.9.2

Generieren der doxygen Dokumentation (optional):

1. Installation von doxygen
2. Befehl:

        doxygen docs/config

## Installation

Zur Installation führt man den 'make' Befehl im Root-Verzeichnis des Projektes aus.

    make

Das cpp2np Modul wird automatisch gebaut und im environment installiert. Im Makefile gibt es noch den unten stehenden Abschnitt. Damit steuert man, ob die .so Datei lokal im Projekt installiert wird, oder global.

    # use for debugging (local)
        pip install -e .
    # use for production (global)
    #	pip install .
    
## Problem mit setup.py

Falls die Installation nicht klappt, könnte eine Lösung sein, die .so Datei manuell zum Pfad
der dynamic linked libraries hinzuzufügen.

Dazu führt man folgenden Befehl im Root-Verzeichnis des Projektes aus:

    echo export LD_LIBRARY_PATH=$(pwd)/build/lib.linux-x86_64-cpython-310 >> ~/.bashrc

## Demo 1: Python(C++)

### Import module

```python
import cpp2np as c2n
import numpy as np
```

### Get pointer to 2x2 std::array allocated by c++:

```python
>>> pointer, shape = c2n.c_arr_i4()
>>> print(pointer)
>>> print(shape)

24245840
(2, 2)
```

### Wrap pointer in numpy array

```python
>>> wrapper = c2n.wrap(pointer, shape, dtype=np.dtype("int32"))
>>> print(wrapper)
>>> print(type(wrapper))

[[1 2]
[3 4]]

<class numpy.ndarray>
```

### Change value in numpy array

```python
>>> wrapper[0,0] = 255
>>> print(wrapper)

[[255   2]
[  3   4]]
```

### Delete numpy array and create new wrapper from same pointer

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

[[24407120        0]
[19943440        0]]
```

We observe that the numpy array is pointing nowhere as the original buffer was freed on the c++ side.

## Demo 2

Apart from the source module in `demo2.py` there's a html file and jupyter notebook in the docs directory containing sample outputs of the script: *docs/demo2.html*.

## Demo 3: C++(Python)

Demonstrates the usage of cpp2np in the other direction. Allocating memory in numpy and using it in C++.

### Create new numpy:

```python
>>> new_arr = np.ones((8,8), dtype="int16")
```

### Retrieve pointer

```python
>>> new_arr_ptr = c2n.descr(new_arr)['data']
>>> print("pointer in python: " + str(new_arr_ptr))

pointer in python: 37148928
```

### Print Numpy array in C++:

```python
>>> c2n.print_arr(new_arr_ptr)

pointer address: 37148928
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
```

The *print_arr* method also leaves its mark by changing the first value to `255`.

### Verify change of value in Python

```python
>>> print(new_arr)

[[255   1   1   1   1   1   1   1]
 [  1   1   1   1   1   1   1   1]
 [  1   1   1   1   1   1   1   1]
 [  1   1   1   1   1   1   1   1]
 [  1   1   1   1   1   1   1   1]
 [  1   1   1   1   1   1   1   1]
 [  1   1   1   1   1   1   1   1]
 [  1   1   1   1   1   1   1   1]]
 ```

 As we see, the memory of the numpy area can be used in C++ and is writable, too.