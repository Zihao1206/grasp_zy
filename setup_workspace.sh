#!/bin/bash
# setup_workspace.sh - Initialize grasp_zy ROS2 workspace
#
# Usage:
#   chmod +x setup_workspace.sh
#   ./setup_workspace.sh
#
# This script:
#   1. Creates workspace structure
#   2. Clones main repository
#   3. Imports external dependencies via vcstool
#   4. Installs system dependencies via rosdep
#   5. Builds the workspace

set -e  # Exit on error

# Configuration
WORKSPACE_NAME="grasp_zy_ws"
REPO_URL="${REPO_URL:-https://github.com/your-org/grasp_zy.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check ROS2
    if [ -z "$ROS_DISTRO" ]; then
        print_error "ROS_DISTRO not set. Please source ROS2 environment first."
        exit 1
    fi
    
    # Check vcstool
    if ! command -v vcs &> /dev/null; then
        print_warn "vcstool not found. Installing..."
        sudo apt update && sudo apt install -y python3-vcstool
    fi
    
    # Check rosdep
    if ! command -v rosdep &> /dev/null; then
        print_warn "rosdep not found. Installing..."
        sudo apt update && sudo apt install -y python3-rosdep
        rosdep init || true
        rosdep update
    fi
    
    # Check colcon
    if ! command -v colcon &> /dev/null; then
        print_warn "colcon not found. Installing..."
        sudo apt update && sudo apt install -y python3-colcon-common-extensions
    fi
    
    print_info "Prerequisites OK"
}

# Create workspace structure
create_workspace() {
    print_info "Creating workspace: ${WORKSPACE_NAME}"
    
    mkdir -p "${WORKSPACE_NAME}/src"
    cd "${WORKSPACE_NAME}"
    
    print_info "Workspace created at: $(pwd)"
}

# Clone main repository
clone_main_repo() {
    print_info "Cloning main repository..."
    
    if [ -d "src/zy_interfaces" ]; then
        print_warn "Repository already cloned (src/zy_interfaces exists). Skipping clone."
    else
        local TEMP_DIR="/tmp/grasp_zy_clone_$$"
        git clone -b "${REPO_BRANCH}" "${REPO_URL}" "${TEMP_DIR}"
        
        for pkg in zy_interfaces zy_camera zy_vision zy_robot zy_comm zy_executor zy_bringup grasp_zy; do
            cp -r "${TEMP_DIR}/${pkg}" src/ 2>/dev/null || true
        done
        cp "${TEMP_DIR}/grasp_zy.repos" . 2>/dev/null || true
        
        rm -rf "${TEMP_DIR}"
        print_info "Repository packages copied to src/"
    fi
}

# Import external dependencies
import_dependencies() {
    print_info "Importing external dependencies..."
    
    if [ -f "grasp_zy.repos" ]; then
        vcs import src < grasp_zy.repos
        print_info "External dependencies imported"
    else
        print_warn "grasp_zy.repos not found. Skipping dependency import."
    fi
}

# Install system dependencies
install_dependencies() {
    print_info "Installing system dependencies via rosdep..."
    
    rosdep install --from-paths src --ignore-src -y --rosdistro "${ROS_DISTRO}" || {
        print_warn "Some dependencies may have failed to install. Check output above."
    }
    
    print_info "Dependencies installed"
}

# Build workspace
build_workspace() {
    print_info "Building workspace..."
    
    colcon build --symlink-install
    
    print_info "Build complete"
}

# Print next steps
print_next_steps() {
    echo ""
    echo "=========================================="
    echo "  Workspace setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Source the workspace:"
    echo "     cd ${WORKSPACE_NAME}"
    echo "     source install/setup.bash"
    echo ""
    echo "  2. Add to .bashrc for auto-sourcing:"
    echo "     echo 'source $(pwd)/install/setup.bash' >> ~/.bashrc"
    echo ""
    echo "  3. Run a test launch:"
    echo "     ros2 launch zy_bringup grasp_system.launch.py"
    echo ""
}

# Main execution
main() {
    print_info "Starting grasp_zy workspace setup..."
    echo ""
    
    check_prerequisites
    create_workspace
    clone_main_repo
    import_dependencies
    install_dependencies
    build_workspace
    print_next_steps
}

# Run main function
main "$@"
