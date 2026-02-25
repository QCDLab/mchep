#!/bin/sh

set -eu

prefix=
version=
target=

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --version VERSION    Specify the version to install (default: latest)"
    echo "  --prefix PATH        Specify the installation prefix"
    echo "  --target TARGET      Specify the target triple (e.g., x86_64-unknown-linux-gnu)"
    echo "  --help               Show this help message"
}

while [ $# -gt 0 ]; do
    case $1 in
        --version)
            version=$2
            shift 2
            ;;
        --version=*)
            version=${1#--version=}
            shift
            ;;
        --prefix)
            prefix=$2
            shift 2
            ;;
        --prefix=*)
            prefix=${1#--prefix=}
            shift
            ;;
        --target)
            target=$2
            shift 2
            ;;
        --target=*)
            target=${1#--target=}
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Error: argument '$1' unknown"
            usage
            exit 1
            ;;
    esac
done

if [ -z "${target}" ]; then
    case $(uname -m):$(uname -s) in
        arm64:Darwin)
            target=aarch64-apple-darwin;;
        x86_64:Darwin)
            target=x86_64-apple-darwin;;
        x86_64:Linux)
            target=x86_64-unknown-linux-gnu;;
        *)
            echo "Error: unknown target, uname = '$(uname -a)'"
            exit 1;;
    esac
fi

# if no prefix is given, prompt for one
if [ -z "${prefix}" ]; then
    printf "Enter installation path: "
    read -r <&1 prefix
    echo
fi

eval mkdir -p "${prefix}"
prefix_abs=$(cd "${prefix}" && pwd)

if [ -z "${version}" ]; then
    version=$(curl -s https://api.github.com/repos/tanjona/mchep/releases/latest | 
        sed -n 's/[ ]*"tag_name"[ ]*:[ ]*"v\([^"]*\)"[ ]*,[ ]*$/\1/p')
    if [ -z "${version}" ]; then
        echo "Error: could not determine the latest version automatically."
        exit 1
    fi
fi

url="https://github.com/tanjona/mchep/releases/download/v${version}/mchep_capi-${target}.tar.gz"

echo "Installation Details:"
echo "  Prefix:  '${prefix_abs}'"
echo "  Target:  '${target}'"
echo "  Version: '${version}'"
echo "  URL:     '${url}'"
echo ""

curl -s -LJ "${url}" | tar xzf - -C "${prefix_abs}"

pc_file="${prefix_abs}/lib/pkgconfig/mchep_capi.pc"
if [ -f "${pc_file}" ]; then
    sed "s:prefix=/:prefix=${prefix_abs}:" "${pc_file}" > "${pc_file}.new"
    mv "${pc_file}.new" "${pc_file}"
fi

pcbin=
if command -v pkg-config >/dev/null; then
    pcbin=$(command -v pkg-config)
elif command -v pkgconf >/dev/null; then
    pcbin=$(command -v pkgconf)
fi

if [ -n "${pcbin}" ]; then
    export PKG_CONFIG_PATH="${prefix_abs}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    if "${pcbin}" --exists mchep_capi; then
        echo "Success: mchep_capi is now available via pkg-config."
    else
        echo "Warning: mchep_capi was installed, but pkg-config cannot find it."
        echo "Make sure to add the following to your shell profile:"
        echo "  export PKG_CONFIG_PATH=${prefix_abs}/lib/pkgconfig:\$PKG_CONFIG_PATH"
    fi
else
    echo "Warning: neither `pkg-config` nor `pkgconf` was found."
    echo "You will need to manually set compiler/linker flags."
fi

echo ""
echo "Installation complete."
