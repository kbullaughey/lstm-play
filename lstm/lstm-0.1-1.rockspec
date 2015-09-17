package = "lstm"
version = "0.1-1"

source = {
  url = '...'
}

description = {
  summary = "Toy LSTM implementation",
  detailed = [[
    LSTM implementation inspired by https://github.com/wojzaremba/lstm
  ]]
}

dependencies = {
  "torch >= 7.0",
  "penlight >= 1.3"
}

build = {
  type = "command",
  build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
  ]],
  install_command = "cd build && $(MAKE) install"
}
