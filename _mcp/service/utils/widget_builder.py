"""Widget builder that compiles widgets on server startup."""

from __future__ import annotations
import subprocess
import pathlib
from _mcp.logger import app_logger as logger


def build_widgets() -> bool:
    """Build all widgets using npm build."""
    widgets_dir = pathlib.Path(__file__).parent.parent.parent / "widgets"
    
    if not widgets_dir.exists():
        logger.error(f"Widgets directory not found: {widgets_dir}")
        return False
    
    logger.info("Building widgets...")
    
    try:
        import platform
        import shutil
        
        is_windows = platform.system() == "Windows"
        
        npm_path = shutil.which("npm")
        if not npm_path:
            if is_windows:
                npm_path = shutil.which("npm.cmd")
            
            if not npm_path:
                logger.error("❌ npm not found in PATH. Please install Node.js")
                logger.error("   Visit https://nodejs.org/ to install Node.js")
                return False
        
        logger.debug(f"📦 Using npm at: {npm_path} (OS: {platform.system()})")
        
        node_modules = widgets_dir / "node_modules"
        package_json = widgets_dir / "package.json"
        
        if package_json.exists() and not node_modules.exists():
            logger.info("📦 Installing widget dependencies...")
            install_result = subprocess.run(
                [npm_path, "install"],
                cwd=str(widgets_dir),
                capture_output=True,
                text=True,
                timeout=300,
                shell=is_windows,
            )
            if install_result.returncode != 0:
                logger.error(f"❌ Failed to install dependencies")
                logger.error(f"stderr: {install_result.stderr}")
                if install_result.stdout:
                    logger.debug(f"stdout: {install_result.stdout}")
                return False
            logger.info("✅ Dependencies installed successfully")
        
        logger.info("🔨 Running build script...")
        result = subprocess.run(
            [npm_path, "run", "build"],
            cwd=str(widgets_dir),
            capture_output=True,
            text=True,
            timeout=300,
            shell=is_windows,
        )
        
        if result.returncode == 0:
            logger.info("✅ Widgets built successfully")
            if result.stdout:
                logger.debug(result.stdout)
            return True
        else:
            logger.error(f"❌ Widget build failed:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Widget build timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to build widgets: {e}")
        return False


def ensure_widgets_built() -> bool:
    """Ensure widgets are built. Build them if necessary."""
    return build_widgets()
