import * as THREE from "three";

export interface RadarNode {
  id: number;
  x: number;
  y: number;
  r: number;
  data: {
    pressure?: number;
    volume?: number;
    solvent?: number;
    solubles?: Record<string, number>;
    local_evidence?: number;
    direction?: string | null;
    is_anchor?: boolean;
    is_center?: boolean;
    network_root?: number | null;
  };
}

export interface RadarLink {
  source: number | RadarNode;
  target: number | RadarNode;
  influence?: {
    from?: number;
    to?: number;
    count?: number;
    maturity?: number;
    avg?: number;
    flow?: number;
    component_flows?: Record<string, number>;
  } | null;
}

export interface TubeState {
  traversal: number[];
  direction: "forward" | "reverse";
  edges: [number, number][];
  solvent: number[];
  solubles: Record<string, number>[];
  pressures: number[];
}

export interface FluxRenderFrame {
  width: number;
  height: number;
  nodes: RadarNode[];
  links: RadarLink[];
  tubes: TubeState[];
}

interface QuadVertexData {
  positions: number[];
  colors: number[];
  along: number[];
  flow: number[];
  indices: number[];
}

const VERTEX_SHADER = `
attribute vec3 color;
attribute float along;
attribute float flow;
varying vec3 vColor;
varying float vAlong;
varying float vFlow;
void main() {
  vColor = color;
  vAlong = along;
  vFlow = flow;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = `
precision highp float;
uniform float time;
uniform float opacity;
varying vec3 vColor;
varying float vAlong;
varying float vFlow;
void main() {
  float moving = fract(vAlong * 7.0 - time * vFlow);
  float pulse = smoothstep(0.02, 0.20, moving) *
                (1.0 - smoothstep(0.30, 0.52, moving));
  vec3 color = vColor * (0.64 + 0.72 * pulse);
  gl_FragColor = vec4(color, opacity);
}
`;

function cssColor(value: string): THREE.Color {
  return new THREE.Color(value);
}

function solubleHue(name: string): number {
  let hash = 2166136261;
  for (let i = 0; i < name.length; i += 1) {
    hash ^= name.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash) % 360;
}

function dominantFluid(
  solvent: number,
  solubles: Record<string, number>,
): THREE.Color {
  let dominantName = "";
  let dominantAmount = 0;
  for (const [name, amount] of Object.entries(solubles)) {
    if (amount > dominantAmount) {
      dominantName = name;
      dominantAmount = amount;
    }
  }
  if (!dominantName || solvent >= dominantAmount) {
    return cssColor("hsl(200, 85%, 64%)");
  }
  return cssColor(`hsl(${solubleHue(dominantName)}, 86%, 61%)`);
}

function nodeColor(node: RadarNode): THREE.Color {
  if (node.data.is_anchor || node.data.is_center) {
    return cssColor("#f0cf65");
  }
  if (node.data.network_root != null) {
    return cssColor(`hsl(${(node.data.network_root * 67) % 360}, 66%, 58%)`);
  }
  return node.data.direction === "backward"
    ? cssColor("#d883d9")
    : cssColor("#66d6a1");
}

function resolveNode(
  endpoint: number | RadarNode,
  lookup: Map<number, RadarNode>,
): RadarNode | undefined {
  return typeof endpoint === "number" ? lookup.get(endpoint) : endpoint;
}

function addQuad(
  data: QuadVertexData,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  halfWidth: number,
  offset: number,
  color: THREE.Color,
  flow: number,
  z: number,
): void {
  const dx = bx - ax;
  const dy = by - ay;
  const length = Math.max(1e-6, Math.hypot(dx, dy));
  const nx = -dy / length;
  const ny = dx / length;
  const ox = nx * offset;
  const oy = ny * offset;
  const wx = nx * halfWidth;
  const wy = ny * halfWidth;
  const base = data.positions.length / 3;
  data.positions.push(
    ax + ox - wx, ay + oy - wy, z,
    ax + ox + wx, ay + oy + wy, z,
    bx + ox + wx, by + oy + wy, z,
    bx + ox - wx, by + oy - wy, z,
  );
  for (let i = 0; i < 4; i += 1) {
    data.colors.push(color.r, color.g, color.b);
    data.flow.push(flow);
  }
  data.along.push(0, 0, 1, 1);
  data.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

function geometryFrom(data: QuadVertexData): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(data.positions, 3),
  );
  geometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(data.colors, 3),
  );
  geometry.setAttribute(
    "along",
    new THREE.Float32BufferAttribute(data.along, 1),
  );
  geometry.setAttribute(
    "flow",
    new THREE.Float32BufferAttribute(data.flow, 1),
  );
  geometry.setIndex(data.indices);
  geometry.computeBoundingSphere();
  return geometry;
}

function emptyQuadData(): QuadVertexData {
  return { positions: [], colors: [], along: [], flow: [], indices: [] };
}

export class FluxWebGLRenderer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.OrthographicCamera;

  private readonly hullMaterial: THREE.ShaderMaterial;
  private readonly lumenMaterial: THREE.ShaderMaterial;
  private readonly nodeMaterial: THREE.MeshBasicMaterial;
  private readonly fluidMaterial: THREE.MeshBasicMaterial;
  private hullMesh: THREE.Mesh | null = null;
  private lumenMesh: THREE.Mesh | null = null;
  private nodeMesh: THREE.InstancedMesh | null = null;
  private fluidMesh: THREE.InstancedMesh | null = null;
  private frame: FluxRenderFrame | null = null;
  private animationHandle = 0;
  private disposed = false;

  constructor(canvas: HTMLCanvasElement) {
    const context = canvas.getContext("webgl2", {
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    if (!context) {
      throw new Error("WebGL2 is unavailable");
    }
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      context,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.camera = new THREE.OrthographicCamera(0, 900, 0, 560, 0.1, 100);
    this.camera.position.z = 20;
    this.camera.lookAt(0, 0, 0);

    this.hullMaterial = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        time: { value: 0 },
        opacity: { value: 0.44 },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
    });
    this.lumenMaterial = this.hullMaterial.clone();
    this.lumenMaterial.uniforms.opacity.value = 0.92;
    this.nodeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.96,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.fluidMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.88,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.animate();
  }

  update(frame: FluxRenderFrame): void {
    this.frame = frame;
    this.resize(frame.width, frame.height);
    this.rebuildHulls(frame);
    this.rebuildLumens(frame);
    this.rebuildNodes(frame);
  }

  private resize(width: number, height: number): void {
    const cssWidth = this.renderer.domElement.clientWidth || width;
    const cssHeight = this.renderer.domElement.clientHeight || height;
    this.renderer.setSize(cssWidth, cssHeight, false);
    this.camera.left = 0;
    this.camera.right = width;
    this.camera.top = 0;
    this.camera.bottom = height;
    this.camera.updateProjectionMatrix();
  }

  private updateQuadMesh(
    current: THREE.Mesh | null,
    data: QuadVertexData,
    material: THREE.Material,
  ): THREE.Mesh {
    const position = current?.geometry.getAttribute("position");
    if (
      current
      && position
      && position.count === data.positions.length / 3
    ) {
      (position.array as Float32Array).set(data.positions);
      (current.geometry.getAttribute("color").array as Float32Array).set(data.colors);
      (current.geometry.getAttribute("along").array as Float32Array).set(data.along);
      (current.geometry.getAttribute("flow").array as Float32Array).set(data.flow);
      position.needsUpdate = true;
      current.geometry.getAttribute("color").needsUpdate = true;
      current.geometry.getAttribute("along").needsUpdate = true;
      current.geometry.getAttribute("flow").needsUpdate = true;
      current.geometry.computeBoundingSphere();
      return current;
    }
    if (current) {
      this.scene.remove(current);
      current.geometry.dispose();
    }
    const mesh = new THREE.Mesh(geometryFrom(data), material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    return mesh;
  }

  private rebuildHulls(frame: FluxRenderFrame): void {
    const lookup = new Map(frame.nodes.map((node) => [node.id, node]));
    const data = emptyQuadData();
    for (const link of frame.links) {
      const source = resolveNode(link.source, lookup);
      const target = resolveNode(link.target, lookup);
      if (!source || !target) continue;
      const influence = link.influence ?? {};
      const count = Math.max(1, Number(influence.count) || 1);
      const maturity = Math.max(0, Math.min(1, Number(influence.maturity) || 0));
      const quality = Math.max(0, Math.min(1, Number(influence.avg) || 0));
      const width = 2.6 + Math.sqrt(count) * 1.6 + maturity * 2.4;
      const color = new THREE.Color().setHSL(quality * 0.32, 0.72, 0.42);
      addQuad(
        data,
        source.x, source.y, target.x, target.y,
        width, 0, color, Number(influence.flow) || 0, 0,
      );
    }
    this.hullMesh = this.updateQuadMesh(
      this.hullMesh,
      data,
      this.hullMaterial,
    );
  }

  private rebuildLumens(frame: FluxRenderFrame): void {
    const lookup = new Map(frame.nodes.map((node) => [node.id, node]));
    const perEdge = new Map<string, Array<{
      tube: TubeState;
      segment: number;
      edge: [number, number];
    }>>();
    for (const tube of frame.tubes) {
      tube.edges.forEach((edge, segment) => {
        const key = `${edge[0]}:${edge[1]}`;
        const list = perEdge.get(key) ?? [];
        list.push({ tube, segment, edge });
        perEdge.set(key, list);
      });
    }

    const data = emptyQuadData();
    for (const entries of perEdge.values()) {
      const laneSpacing = Math.min(
        1.7,
        12 / Math.max(1, entries.length - 1),
      );
      entries.forEach(({ tube, segment, edge }, lane) => {
        const source = lookup.get(edge[0]);
        const target = lookup.get(edge[1]);
        if (!source || !target) return;
        const laneOffset = (
          lane - (entries.length - 1) / 2
        ) * laneSpacing;
        const solvent = Number(tube.solvent[segment]) || 0;
        const solubles = tube.solubles[segment] ?? {};
        const pressure = Number(tube.pressures[segment]) || 0;
        const amount = solvent + Object.values(solubles).reduce(
          (sum, value) => sum + Number(value || 0), 0,
        );
        const flow = (tube.direction === "forward" ? 1 : -1)
          * (0.25 + Math.min(4, pressure + amount));
        addQuad(
          data,
          source.x, source.y, target.x, target.y,
          Math.min(
            0.58 + Math.min(1.4, Math.sqrt(Math.max(0, amount)) * 0.32),
            Math.max(0.12, laneSpacing * 0.38),
          ),
          laneOffset,
          dominantFluid(solvent, solubles),
          flow,
          2,
        );
      });
    }
    this.lumenMesh = this.updateQuadMesh(
      this.lumenMesh,
      data,
      this.lumenMaterial,
    );
  }

  private rebuildNodes(frame: FluxRenderFrame): void {
    const count = frame.nodes.length;
    if (!this.nodeMesh || this.nodeMesh.count !== count) {
      if (this.nodeMesh) {
        this.scene.remove(this.nodeMesh);
        this.nodeMesh.geometry.dispose();
      }
      if (this.fluidMesh) {
        this.scene.remove(this.fluidMesh);
        this.fluidMesh.geometry.dispose();
      }
      const geometry = new THREE.CircleGeometry(1, 24);
      this.nodeMesh = new THREE.InstancedMesh(
        geometry, this.nodeMaterial, count,
      );
      this.fluidMesh = new THREE.InstancedMesh(
        geometry.clone(), this.fluidMaterial, count,
      );
      this.nodeMesh.frustumCulled = false;
      this.fluidMesh.frustumCulled = false;
      this.scene.add(this.nodeMesh, this.fluidMesh);
    }
    const maxVolume = Math.max(
      1e-9,
      ...frame.nodes.map((node) => Number(node.data.volume) || 0),
    );
    const nodeMesh = this.nodeMesh;
    const fluidMesh = this.fluidMesh;
    if (!nodeMesh || !fluidMesh) return;
    const matrix = new THREE.Matrix4();
    frame.nodes.forEach((node, index) => {
      matrix.makeTranslation(node.x, node.y, 4);
      matrix.scale(new THREE.Vector3(node.r, node.r, 1));
      nodeMesh.setMatrixAt(index, matrix);
      nodeMesh.setColorAt(index, nodeColor(node));

      const volume = Math.max(0, Number(node.data.volume) || 0);
      const fluidScale = node.r * Math.min(0.9, Math.sqrt(volume / maxVolume));
      matrix.makeTranslation(node.x, node.y, 5);
      matrix.scale(new THREE.Vector3(fluidScale, fluidScale, 1));
      fluidMesh.setMatrixAt(index, matrix);
      fluidMesh.setColorAt(
        index,
        dominantFluid(
          Number(node.data.solvent) || 0,
          node.data.solubles ?? {},
        ),
      );
    });
    nodeMesh.instanceMatrix.needsUpdate = true;
    fluidMesh.instanceMatrix.needsUpdate = true;
    if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true;
    if (fluidMesh.instanceColor) fluidMesh.instanceColor.needsUpdate = true;
  }

  private animate = (): void => {
    if (this.disposed) return;
    const elapsed = performance.now() / 1000;
    this.hullMaterial.uniforms.time.value = elapsed;
    this.lumenMaterial.uniforms.time.value = elapsed;
    this.renderer.render(this.scene, this.camera);
    this.animationHandle = requestAnimationFrame(this.animate);
  };

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.animationHandle);
    this.hullMesh?.geometry.dispose();
    this.lumenMesh?.geometry.dispose();
    this.nodeMesh?.geometry.dispose();
    this.fluidMesh?.geometry.dispose();
    this.hullMaterial.dispose();
    this.lumenMaterial.dispose();
    this.nodeMaterial.dispose();
    this.fluidMaterial.dispose();
    this.renderer.dispose();
  }
}
