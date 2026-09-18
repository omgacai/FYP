import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';
import {localStore} from './localStore.js';
const root=fileURLToPath(new URL('../cubicasa_eval/',import.meta.url));
export default defineConfig({server:{host:'127.0.0.1'},plugins:[{name:'local-annotation-storage',configureServer(server){server.middlewares.use(localStore(root));},configurePreviewServer(server){server.middlewares.use(localStore(root));}}]});
