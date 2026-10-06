import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';
import {evaluationStore} from './evaluationStore.js';
import {localStore} from './localStore.js';
import {qaStore} from './qaStore.js';
// The editor is used to review the checked-in CubiCasa benchmark directly.
// Override with FYP_ANNOTATION_ROOT when working on a separate copy.
const root=process.env.FYP_ANNOTATION_ROOT || fileURLToPath(new URL('../cubicasa_benchmark/',import.meta.url));
const qaRoot=process.env.FYP_QA_RESULTS_ROOT || fileURLToPath(new URL('../cluster_results/manual20_qa/',import.meta.url));
export default defineConfig({cacheDir:'.vite-cache',server:{host:'127.0.0.1'},plugins:[{name:'local-annotation-storage',configureServer(server){server.middlewares.use(qaStore(root,qaRoot));server.middlewares.use(evaluationStore(root));server.middlewares.use(localStore(root));},configurePreviewServer(server){server.middlewares.use(qaStore(root,qaRoot));server.middlewares.use(evaluationStore(root));server.middlewares.use(localStore(root));}}]});
