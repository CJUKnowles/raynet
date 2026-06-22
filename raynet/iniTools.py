"""INI generation helpers for RayNet simulations."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


def as_bool(value, default=False):
    """Parse common bool-like values used in runner configs."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def placeholder_key(key):
    """Return the INI placeholder for a user-facing replacement key."""
    key = str(key)
    if key.startswith("!") and key.endswith("!"):
        return key
    return f"!{key.upper()}!"


def normalize_replacements(replacements):
    """Convert friendly replacement names to INI placeholder keys."""
    return {
        placeholder_key(key): str(value)
        for key, value in (replacements or {}).items()
        if value is not None
    }


def normalize_overrides(overrides):
    """Convert line-key overrides to strings."""
    return {
        str(key): str(value)
        for key, value in (overrides or {}).items()
        if value is not None
    }


def replace_placeholders(text, replacements):
    """Apply ``!PLACEHOLDER!``-style replacements to INI text."""
    for key, value in normalize_replacements(replacements).items():
        text = text.replace(key, value)
    return text


def apply_line_overrides(text, overrides):
    """Replace values for INI assignment lines matching each override key."""
    for key, value in normalize_overrides(overrides).items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
        replacement = rf"\g<1>{value}"
        text, count = pattern.subn(replacement, text)
        if count == 0:
            text = text.rstrip() + f"\n\n{key} = {value}\n"
    return text


def default_replacements(*, raynet_path=None, home=None):
    """Return common RayNet INI template replacements."""
    return {
        "home": home or os.environ.get("HOME", str(Path.home())),
        "raynet_path": str(raynet_path or os.environ.get("RAYNET_PATH", Path.cwd())),
    }



class IniWrapper:
    """Manage one generated INI file derived from a template."""

    def __init__(
        self,
        template_path,
        replacements=None,
        *,
        overrides=None,
        directory=None,
        prefix=None,
        suffix=".ini",
    ):
        self.template_path = Path(template_path).expanduser()
        if not self.template_path.exists():
            raise FileNotFoundError(f"ini_path not found: {ini_path}")
        self.replacements = normalize_replacements(replacements)
        self.overrides = normalize_overrides(overrides)
        self.directory = Path(directory).expanduser() if directory else None
        self.prefix = prefix or f"{self.template_path.stem}_"
        self.suffix = suffix
        self.generated_path = None

    def add_replacements(self, replacements=None, **kwargs):
        """Merge additional placeholder replacements."""
        updates = {}
        updates.update(replacements or {})
        updates.update(kwargs)
        self.replacements.update(normalize_replacements(updates))
        return self

    def add_replacement(self, key, value):
        """Add one placeholder replacement."""
        self.replacements[placeholder_key(key)] = str(value)
        return self

    def add_overrides(self, overrides=None, **kwargs):
        """Merge additional line-key overrides."""
        updates = {}
        updates.update(overrides or {})
        updates.update(kwargs)
        self.overrides.update(normalize_overrides(updates))
        return self

    def add_override(self, key, value):
        """Add one line-key override."""
        self.overrides[str(key)] = str(value)
        return self

    def render_text(self):
        """Render the generated INI text without writing it."""
        text = self.template_path.read_text(encoding="utf-8")
        text = replace_placeholders(text, self.replacements)
        return apply_line_overrides(text, self.overrides)

    def _default_directory(self):
        directory = self.directory or self.template_path.parent / "ini_variants"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def materialize_ini(self, *, output_path=None, directory=None, prefix=None, suffix=None):
        """Generate the concrete INI file and return its path."""
        text = self.render_text()
        if output_path is not None:
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        else:
            target_dir = Path(directory).expanduser() if directory else self._default_directory()
            target_dir.mkdir(parents=True, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                prefix=prefix if prefix is not None else self.prefix,
                suffix=suffix if suffix is not None else self.suffix,
                dir=str(target_dir),
                delete=False,
            )
            try:
                tmp.write(text)
                path = Path(tmp.name)
            finally:
                tmp.close()
        self.generated_path = path
        return path

    def cleanup(self):
        """Delete the generated INI file, if one exists."""
        if self.generated_path is None:
            return
        try:
            self.generated_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.generated_path = None


iniWrapper = IniWrapper
