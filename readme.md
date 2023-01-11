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

Demonstrates the usage of cpp2np in the other direction. Allocating memory in numpy, then access and use it in C++.

### Create regular numpy array:

```python
>>> new_arr = np.ones((8,8), dtype="uint8")
```

### Retrieve pointer

```python
>>> ptr = c2n.descr(new_arr)['data']
>>> print("pointer in python: " + str(ptr))

pointer in python: 35841504
```


### Disable OWNDATA flag

By disabling this flag we prevent the numpy array from the deleting the memory while we are still using it in C++.

```python
>>> c2n.owndata(new_arr, False)
```

### Test if disabling the flag worked

Now we delete the numpy array and check if the memory area is still there. Before we print, we do some allocations of new memory blocks which would likely cause the original data area to get overriden if it had been freed.

```python
>>> del new_arr
# do some stuff in memory 
>>> a = np.zeros((10,10), dtype="double")
>>> b = np.zeros((5,5))
# now print memory the pointer is referencing
>>> print("\nprint numpy data from c++:")
>>> c2n.print_testarr(ptr)

print numpy data from c++:
pointer address: 35841504
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
```

Thankfully we can see that the memory still exists, even after deletion of the numpy array.


### Free numpy allocated data from C++:

When the c++ object is done using the memory, it can use `cpp2np_py_free` to free the data. Note that this is a different method than `cpp2np_free`, as Python has its own memory management.

```python
>>> c2n.py_free(ptr)
```

Now we can check again to see if we can still access the memory from c++:

```python
# some memory stuff happening again
>>> a = np.zeros((10,10), dtype="double")
>>> b = np.zeros((5,5))
# try to print data
>>> c2n.print_testarr(ptr)

pointer address: 35841504
[   0    0    0    0    0    0    0    0]
[  16   96  226    1    0    0    0    0]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
[   1    1    1    1    1    1    1    1]
```

We see the data area was overwritten by other objects in memory, which means it got freed by our method.