## Code Translation

The file tree below describes the structure of the translation data. It is divided into two directories, ```pair_data_tok_1``` for snippet level translation and ```pair_data_tok_full``` for program level translation. Each sub task contains 21 directories for each possible language pair. The same pairs can be used with the ```src``` and ```trg``` languages switched to perform the experiments for the opposing pairs.

```
XLCoST_data
|
├── pair_data_tok_1
│   ├── C++-C
│   ├── C++-C#
│   ├── C++-Javascript
│   ├── C++-PHP
│   ├── C++-Python
│   ├── C-C#
│   ├── C-Javascript
│   ├── C-PHP
│   ├── C-Python
|   .
|   .
|   .
|   .
|   21 directories
|   
├── pair_data_tok_full
    ├── C++-C
    ├── C++-C#
    ├── C++-Javascript
    ├── C++-PHP
    ├── C++-Python
    ├── C-C#
    ├── C-Javascript
    ├── C-PHP
    ├── C-Python
    .
    .
    .
    .
    21 directories
```

Each language pair directory contains the ```train``` ```test``` and ```val``` splits of the code data. The ```map.jsonl``` files contain a line by line mapping of the ```problem_id``` to the lines in the corresponding code files. These relationships and file structures remain exactly the same for ```pair_data_tok_full``` (program level) translation, the only difference being that instead of code snippets, they contain full programs.
```
pair_data_tok_1
├── C++-C
    ├── test-C++-C-tok.c
    ├── test-C++-C-tok.cpp
    ├── test-C++-map.jsonl
    ├── test-C-map.jsonl
    ├── train-C++-C-tok.c
    ├── train-C++-C-tok.cpp
    ├── train-C++-map.jsonl
    ├── train-C-map.jsonl
    ├── val-C++-C-tok.c
    ├── val-C++-C-tok.cpp
    ├── val-C++-map.jsonl
    └── val-C-map.jsonl
```
### Data Samples

```train-C++-C-tok.cpp``` file:

```cpp
#include <bits/stdc++.h> NEW_LINE using namespace std ;
void count_setbit ( int N ) {
int result = 0 ;
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( 1 << i ) & N ) {
result ++ ; } } cout << result << endl ; }
```
```train-C++-map.jsonl``` file:
```c
10113-C++-1
10113-C++-2
10113-C++-3
10113-C++-4
10113-C++-5
10113-C++-6
```
`train-C++-C-tok.c` file:
```c
#include <stdio.h>
void count_setbit ( int N ) {
int result = 0 ;
for ( int i = 0 ; i < 32 ; i ++ ) {
if ( ( 1 << i ) & N ) {
result ++ ; } } printf ( " % d STRNEWLINE " , result ) ; }
```

```train-C-map.jsonl``` file:
```c
10113-C-1
10113-C-2
10113-C-3
10113-C-4
10113-C-5
```
