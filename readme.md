# cpp2np

## Requirements
    Python >= 3.9
    numpy >= 1.23.4
    g++ >= 9.3.0
    GNU Make >= 4.2.1

## Installation
    Zur Installation führt man den 'make' Befehl im Root-Verzeichnis des Projektes aus.
    >>> make
    Das cpp2np Modul wird automatisch gebaut und im environment installiert.
    
## Problem mit setup.py
    Falls die Installation nicht klappt, könnte eine Lösung sein, die .so Datei manuell zum Pfad
    der dynamic linked libraries hinzuzufügen.

    Dazu führt man folgenden Befehl aus:

    >>> echo export LD_LIBRARY_PATH=$(pwd)/build/lib.linux-x86_64-cpython-310 >> ~/.bashrc
    