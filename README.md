# OpenCL-WorkShop

Material for the OpenCL Workshop taught by me.

Environment setup:

Download Premake4 binary from https://premake.github.io/download.html#v4 and copy it to a bin folder (Ex.: /usr/local/bin) or make sure its containing folder is added to the PATH.

Clone the oclc repo (https://github.com/lighttransport/oclc) and follow the instructions in the README file in order to build it. 
Add the oclc binary to a bin folder (Ex.: /usr/local/bin) or make sure its containing folder is added to the PATH.

To check OpenCL code do:

oclc program.cl

To build programs do:

On OSX:

gcc program.c -o bin/program -framework OpenCL

Try building the following project to check your setup:

https://github.com/jpaulorio/password_crack
