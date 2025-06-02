# Function to install grep using available package manager
install_grep() {
    if command -v apt-get &> /dev/null; then
        apt-get update -qq && apt-get install -y -qq grep > /dev/null 2>&1
    elif command -v apk &> /dev/null; then
        apk add --no-cache grep > /dev/null 2>&1
    else
        printf "Error: No supported package manager found to install grep\n" >&2
        return 1
    fi
}

# Function to return a items version
get_version() {
    # Check if grep is available
    if ! command -v grep &> /dev/null; then
        printf "Debug: grep not found, attempting to install\n" >&2
        if ! install_grep; then
            printf "Error: Failed to install grep\n" >&2
            exit 1
        fi
        printf "Debug: grep installed successfully\n" >&2
    fi
    # Get version from toml file
    version=$(grep -E -o '^version = *"[^"]+"' pyproject.toml | cut -d'"' -f2)
    printf "Debug: Version is ${version}" >&2
    # Return version
    echo "${version}"
}