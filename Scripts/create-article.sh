#!/bin/bash

# Check if article name is provided
if [ -z "$1" ]; then
    echo "Error: Article name is required"
    echo "Usage: $0 <article-name>"
    exit 1
fi

ARTICLE_NAME="$1"

# Get current year and month
YEAR=$(date +%Y)
MONTH=$(date +%m)

# Create directory path
DIR="Articles/$YEAR/$MONTH"

# Create directory if it doesn't exist
if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
    echo "Created directory: $DIR"
fi

# Create the markdown file
FILE_PATH="$DIR/$ARTICLE_NAME.md"
touch "$FILE_PATH"

echo "Created article: $FILE_PATH"
