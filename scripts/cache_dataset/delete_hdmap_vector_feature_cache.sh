#!/usr/bin/env bash
set -Eeuo pipefail

# 递归清理数据集缓存中的指定文件。
#
# 默认只预览，不删除任何文件。必须显式传入 --delete 才会删除；删除前还会
# 要求输入 DELETE 确认。自动化场景可以使用 --delete --yes。
#
# 脚本只读取配置文件中未被注释的 cache_path 字段。每个 cache_path 下只匹配
# 精确文件名，不会删除目录或同目录中的其他缓存文件。
#
# 常用用法（在仓库根目录执行）：
#   bash scripts/cache_dataset/delete_hdmap_vector_feature_cache.sh
#   bash scripts/cache_dataset/delete_hdmap_vector_feature_cache.sh --delete
#   bash scripts/cache_dataset/delete_hdmap_vector_feature_cache.sh --delete --yes
#   bash scripts/cache_dataset/delete_hdmap_vector_feature_cache.sh \
#       --config navsim/planning/script/config/common/mogo_dataset/nav_station.yaml
#   bash scripts/cache_dataset/delete_hdmap_vector_feature_cache.sh \
#       --name another_cache_file.gz --delete
#
# 参数：
#   -c, --config PATH   manifest/YAML 文件；默认使用 MOGO_MANIFEST，未设置时使用
#                       本仓库的 nav_station.yaml。
#   -n, --name FILE     要匹配的精确文件名；默认 hdmap_vector_feature_v2.gz。
#       --delete        执行删除；不加此参数时只读预览。
#   -y, --yes           跳过 DELETE 二次确认；必须与 --delete 一起使用。
#   -h, --help          显示帮助。
#
# 注意：删除不会清理因此变空的父目录，也不会修改 YAML；缓存通常需要重新生成。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/navsim/planning/script/config/common/mogo_dataset/nudge.yaml"
CONFIG_FILE="${MOGO_MANIFEST:-${DEFAULT_CONFIG}}"
TARGET_NAME="hdmap_vector_feature_v2.gz"
MODE="preview"
ASSUME_YES=false
PREVIEW_LIMIT=20

usage() {
    sed -n '1,75p' "${BASH_SOURCE[0]}"
}

die() {
    printf '错误：%s\n' "$*" >&2
    exit 2
}

while (($# > 0)); do
    case "$1" in
        -c|--config)
            (($# >= 2)) || die "${1} 需要一个 YAML 文件路径"
            CONFIG_FILE="$2"
            shift 2
            ;;
        -n|--name)
            (($# >= 2)) || die "${1} 需要一个文件名"
            TARGET_NAME="$2"
            shift 2
            ;;
        --delete)
            MODE="delete"
            shift
            ;;
        -y|--yes)
            ASSUME_YES=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数：$1（使用 --help 查看用法）"
            ;;
    esac
done

[[ -f "${CONFIG_FILE}" ]] || die "找不到配置文件：${CONFIG_FILE}"
[[ -n "${TARGET_NAME}" ]] || die "目标文件名不能为空"
case "${TARGET_NAME}" in
    */*|*\**|*\?*|*\[*|*\]*)
        die "--name 必须是单个精确文件名，不能包含路径或通配符：${TARGET_NAME}"
        ;;
esac
if [[ "${ASSUME_YES}" == true && "${MODE}" != delete ]]; then
    die "--yes 必须与 --delete 一起使用"
fi

CONFIG_DIR="$(cd -- "$(dirname -- "${CONFIG_FILE}")" && pwd)"

# 提取 active cache_path。当前 manifest 使用简单标量路径，因此无需 PyYAML。
mapfile -t RAW_CACHE_PATHS < <(
    awk '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*cache_path:[[:space:]]*/ {
            value = $0
            sub(/^[[:space:]]*cache_path:[[:space:]]*/, "", value)
            sub(/[[:space:]]+#.*$/, "", value)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            if (substr(value, 1, 1) == "\"" && substr(value, length(value), 1) == "\"") {
                value = substr(value, 2, length(value) - 2)
            } else if (substr(value, 1, 1) == "\047" && substr(value, length(value), 1) == "\047") {
                value = substr(value, 2, length(value) - 2)
            }
            if (value != "") print value
        }
    ' "${CONFIG_FILE}"
)
((${#RAW_CACHE_PATHS[@]} > 0)) || die "配置文件中没有找到 active cache_path：${CONFIG_FILE}"

# 去重，避免 manifest 重复列出同一目录时重复扫描。
declare -A SEEN_CACHE_PATHS=()
CACHE_PATHS=()
for cache_path in "${RAW_CACHE_PATHS[@]}"; do
    if [[ "${cache_path}" != /* ]]; then
        cache_path="${CONFIG_DIR}/${cache_path}"
    fi
    if [[ -z "${SEEN_CACHE_PATHS[${cache_path}]+x}" ]]; then
        SEEN_CACHE_PATHS["${cache_path}"]=1
        CACHE_PATHS+=("${cache_path}")
    fi
done

# cache_path 下的 token 目录可能是指向上游数据集缓存的软链（例如 nudge 这类
# 派生数据集）。因此所有遍历都必须用 find -L 跟随软链，否则匹配数恒为 0。
count_matches() {
    local cache_dir="$1"
    find -L "${cache_dir}" -type f -name "${TARGET_NAME}" -printf '.' 2>/dev/null | wc -c
}

show_sample() {
    local cache_dir="$1"
    local count="$2"
    ((count > 0)) || return 0
    find -L "${cache_dir}" -type f -name "${TARGET_NAME}" -print 2>/dev/null \
        | sed -n "1,${PREVIEW_LIMIT}p"
    if ((count > PREVIEW_LIMIT)); then
        printf '  ...（仅显示前 %d 个，共 %d 个）\n' "${PREVIEW_LIMIT}" "${count}"
    fi
}

printf '配置文件：%s\n' "${CONFIG_FILE}"
printf '目标文件名：%s\n' "${TARGET_NAME}"
printf 'active cache_path 数量：%d\n' "${#CACHE_PATHS[@]}"
printf '操作模式：%s\n\n' "${MODE}"

TOTAL_FOUND=0
EXISTING_DIRS=0
for cache_dir in "${CACHE_PATHS[@]}"; do
    if [[ ! -d "${cache_dir}" ]]; then
        printf '跳过（目录不存在）：%s\n' "${cache_dir}" >&2
        continue
    fi
    EXISTING_DIRS=$((EXISTING_DIRS + 1))
    match_count="$(count_matches "${cache_dir}")"
    TOTAL_FOUND=$((TOTAL_FOUND + match_count))
    printf '[%d 个] %s\n' "${match_count}" "${cache_dir}"
    show_sample "${cache_dir}" "${match_count}"
done

printf '\n匹配总数：%d\n' "${TOTAL_FOUND}"
printf '实际存在的 cache_path：%d/%d\n' "${EXISTING_DIRS}" "${#CACHE_PATHS[@]}"

if [[ "${MODE}" != delete ]]; then
    printf '\n当前为预览模式，未删除任何文件。确认范围后加 --delete 执行。\n'
    exit 0
fi
if ((TOTAL_FOUND == 0)); then
    printf '\n没有找到匹配文件，无需删除。\n'
    exit 0
fi

if [[ "${ASSUME_YES}" != true ]]; then
    printf '\n⚠️ 将删除上述 %d 个文件；此操作不可直接撤销。\n' "${TOTAL_FOUND}"
    printf '请输入 DELETE 以继续：'
    read -r confirmation
    [[ "${confirmation}" == DELETE ]] || die "未输入 DELETE，已取消，未删除任何文件"
fi

# GNU find 拒绝 `-L ... -delete`，所以这里用 -print0 | xargs rm。软链 token 目录下
# 命中的是上游真实文件，删除会作用于上游；软链本身不会被删除。
TOTAL_DELETED=0
for cache_dir in "${CACHE_PATHS[@]}"; do
    [[ -d "${cache_dir}" ]] || continue
    before_count="$(count_matches "${cache_dir}")"
    find -L "${cache_dir}" -type f -name "${TARGET_NAME}" -print0 2>/dev/null \
        | xargs -0 --no-run-if-empty rm -f
    after_count="$(count_matches "${cache_dir}")"
    deleted_count=$((before_count - after_count))
    TOTAL_DELETED=$((TOTAL_DELETED + deleted_count))
    printf '已删除 [%d 个] %s\n' "${deleted_count}" "${cache_dir}"
done

REMAINING=0
for cache_dir in "${CACHE_PATHS[@]}"; do
    [[ -d "${cache_dir}" ]] || continue
    remaining_count="$(count_matches "${cache_dir}")"
    REMAINING=$((REMAINING + remaining_count))
done

printf '\n删除总数：%d\n' "${TOTAL_DELETED}"
printf '删除后剩余匹配：%d\n' "${REMAINING}"
if ((REMAINING != 0)); then
    die "仍有文件未删除，请检查权限或正在写入缓存的进程"
fi
