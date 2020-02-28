#!/bin/bash

# gcc >= 4.7.0

gcc -mavx2 -O3 -fomit-frame-pointer -Wall -static main.c  -g poly1305/poly1305_avx2.c -o poly1305-avx2

