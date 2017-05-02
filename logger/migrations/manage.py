#!/usr/bin/env python
from migrate.versioning.shell import main

if __name__ == '__main__':
    db = 'testdb2'
    main(url='postgresql://colingaudreau:Ee45dij7@130.179.130.0:5432/mastersdb', debug='False', repository='migrations')
