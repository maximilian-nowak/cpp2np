# cpp2np Modul

## Requirements
    Python >= 3.9
    numpy >= 1.23.4
    g++ >= 9.3.0
    GNU Make >= 4.2.1

## Installation
    Zur Installation führt man den 'make' Befehl im Root-Verzeichnis des Projektes aus.
    >>> make
    Das cpp2np Modul wird automatisch gebaut und im environment installiert.
    
## Anmerkungen:
    Die Übergabe der Speicherverwaltung funktioniert nicht nicht ganz.
    Es gibt zwei Optionen:
        - Setzen des 'OWNDATA' flags im array, wodurch das numpy array als Besitzer des Speicherbereichs wird.
        - Einbetten in eine 'capsule', welche als Destruktor fungiert, sobald das numpy array gelöscht wird.
    
    Im Moment bekomme ich noch jedes Mal den Fehler "free(): invalid pointer" vpm Python Interpreter.
    Laut der verlinkten Diskussion in Stackoverflow könnte dies allerdings einfach daran liegen, dass
    ich mein Test-C-Array in Python angelegt habe, und dadurch nicht durch "free" freigegeben werden kann.
    
    https://stackoverflow.com/questions/27529857/invalid-pointer-error-when-using-free
    