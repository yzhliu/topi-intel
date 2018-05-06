#!/usr/bin/env bash
for f in `ls val`
do
    mv val/$f/* val/
done