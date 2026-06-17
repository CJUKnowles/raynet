# RayNet Codex Notes

- Always activate the project environment before running Python commands:
  `source .venv/bin/activate`
- The virtualenv activation script provides required Python modules and custom RayNet environment variables.
- Always build this project with:
  `./build.sh`
- Do not use direct component builds such as `make -C simlibs/...` for verification unless the user explicitly asks for that narrow diagnostic.
