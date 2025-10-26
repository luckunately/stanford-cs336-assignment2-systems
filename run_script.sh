flags=""
# flags="--torch_compile"
# flags="--autocast"
# flags="--autocast --torch_compile"

# Generate filename from flags
filename=$(echo "$flags" | sed 's/--//g; s/ /_/g').txt
# if there is no flags, set filename to no_flags.txt
if [ -z "$flags" ]; then
    filename="no_flags.txt"
fi

uv run python cs336_systems/naive_meassure.py $flags | tee "$filename"