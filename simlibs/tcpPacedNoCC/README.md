# tcpPacedNoCC
A small extension of cubic (https://github.com/Avian688/cubic) that adds congestion control-free option. This is useful for clean-slate schemes that want to make use of pacing, like Orca's CleanSlate. 
- This is a quick-and-dirty approach, but it works just fine.
- A better approach would be to directly extend tcpPaced (https://github.com/Avian688/tcpPaced) and create the a noCC scheme from scratch.