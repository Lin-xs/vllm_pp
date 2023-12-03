if [ -z "$1" ]; then
    echo "ERROR: Need commit message."
else
    git add .
    git commit -m "${1}"
    git push -u origin main
fi