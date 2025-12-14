module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "pip install --upgrade pip wheel setuptools"
      }
    },
    {
      method: "shell.run", 
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "{{gpu === 'nvidia' ? 'pip install torch==2.4.0 torchvision==0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cu121' : 'pip install torch==2.4.0 torchvision==0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cpu'}}"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "{{gpu === 'nvidia' ? 'pip install natten==0.17.1+torch240cu121 -f https://shi-labs.com/natten/wheels/' : 'pip install natten -f https://shi-labs.com/natten/wheels/'}}"
      }
    }
  ]
}
