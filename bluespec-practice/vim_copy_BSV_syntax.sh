#!/bin/sh
# Copies vim BSV syntax highlighting files to the users .vim folder
cd /mit/6.175/misc/vim
mkdir -p ~/.vim
cp -r * ~/.vim
