#!/bin/sh

# This script loads Toulbar from # https://mulcyber.toulouse.inra.fr/projects/toulbar2/
# and applies a patch to make TRW-S-LIB workable with openGM.
#
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
TOULBAR_FILENAME=toulbar2.0.9.7.0-Release-sources.tar.gz
TOULBAR_URL=https://mulcyber.toulouse.inra.fr/frs/download.php/1380/
TOULBAR_SOURCE_FOLDER=../../ToulBar2.src-patched/
TOULBAR_PATCH_NAME=ToulBar2.patch

# check if destination folder already exists
if [ -e "$TOULBAR_SOURCE_FOLDER" ]
then
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download ToulBar2
echo "Getting $TOULBAR_FILENAME from $TOULBAR_URL ..."
if [ -e "$ZIP_FOLDER$TOULBAR_FILENAME" ]
then
    echo "$TOULBAR_FILENAME already exists, skipping download."
else
    wget -q "$TOULBAR_URL$TOULBAR_FILENAME" -P "$ZIP_FOLDER"
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$TOULBAR_FILENAME" ]
then :
else
    echo "Couldn't download $TOULBAR_FILENAME. Check if $TOULBAR_URL$TOULBAR_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $TOULBAR_FILENAME"
mkdir -p "$TOULBAR_SOURCE_FOLDER"
tar --strip=1 -C "$TOULBAR_SOURCE_FOLDER" -xzf "$ZIP_FOLDER$TOULBAR_FILENAME"
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $TOULBAR_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d "$TOULBAR_SOURCE_FOLDER" -p1 < "$PATCH_FOLDER$TOULBAR_PATCH_NAME" -N -r -
if [ "$?" = "0" ]
then
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi
