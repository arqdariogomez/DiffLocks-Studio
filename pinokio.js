
const os = require('os')
const fs = require('fs')
const path = require('path')

module.exports = {
  title: "DiffLocks Studio",
  description: "AI-powered 3D hair generation from a single image",
  icon: "icon.png",
  menu: async (kernel) => {
    // Detectar si ya está instalado verificando si existe el venv
    const installed = await kernel.exists(__dirname, "venv")
    if (installed) {
      return [{
        html: '<i class="fa-solid fa-play"></i> Start',
        href: "pinokio/start.json"
      }, {
        html: '<i class="fa-solid fa-rotate"></i> Update',
        href: "pinokio/update.json"
      }, {
        html: '<i class="fa-solid fa-plug"></i> Re-install',
        href: "pinokio/install.json"
      }]
    } else {
      return [{
        html: '<i class="fa-solid fa-download"></i> Install',
        href: "pinokio/install.json"
      }]
    }
  }
}
