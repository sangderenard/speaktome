export interface LatentRadarNode {
  id: number;
  data: {
    network_root?: number | null;
    direction?: string | null;
    level?: number | null;
    depth?: number;
    pressure?: number;
    volume?: number;
    solvent?: number;
    solubles?: Record<string, number>;
    local_evidence?: number;
    path_mean?: number;
    backward_growth_interest?: number;
    forward_growth_interest?: number;
    is_anchor?: boolean;
    is_center?: boolean;
    latent_coordinates?: number[];
    hyperspherical_angles?: number[];
  };
}

export interface ProjectedObservationNode {
  id: number;
  network: string;
  latent: number[];
  unit: number[];
  projected: [number, number, number];
  shell: [number, number, number];
  confidence: number;
  screen: { x: number; y: number; depth: number; scale: number };
}

export interface HypersphereObservationFrame {
  version: 1;
  dimensions: string[];
  projection: number[][];
  networkCaps: Record<string, [number, number, number]>;
  shellRadius: number;
  nodes: ProjectedObservationNode[];
}

const EPSILON = 1e-9;
const SHARED_DIMENSIONS = [
  "shared:anchor",
  "shared:direction",
  "shared:level",
  "shared:pressure",
  "shared:hydration",
  "shared:ion_load",
  "shared:local_evidence",
  "shared:path_mean",
];
const NETWORK_FEATURES = [
  "membership",
  "pressure",
  "hydration",
  "ion_load",
  "forward_commitment",
  "backward_commitment",
];

function finite(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function networkKey(node: LatentRadarNode): string {
  return node.data.network_root == null
    ? "main"
    : `net:${node.data.network_root}`;
}

function hashText(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function pseudoRandom(seed: number): () => number {
  let state = seed || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) / 0xffffffff) * 2 - 1;
  };
}

function dot(left: number[], right: number[]): number {
  let value = 0;
  for (let index = 0; index < left.length; index += 1) {
    value += left[index] * right[index];
  }
  return value;
}

function normalize(vector: number[]): number[] {
  const length = Math.sqrt(dot(vector, vector));
  if (length <= EPSILON) {
    const fallback = new Array(vector.length).fill(0);
    fallback[0] = 1;
    return fallback;
  }
  return vector.map((value) => value / length);
}

export function hypersphericalToCartesian(angles: number[]): number[] {
  const coordinates = new Array(angles.length + 1).fill(0);
  let sineProduct = 1;
  for (let index = 0; index < angles.length; index += 1) {
    const angle = finite(angles[index]);
    coordinates[index] = sineProduct * Math.cos(angle);
    sineProduct *= Math.sin(angle);
  }
  coordinates[angles.length] = sineProduct;
  return coordinates;
}

function intrinsicCoordinates(node: LatentRadarNode): number[] {
  if (Array.isArray(node.data.latent_coordinates)) {
    return node.data.latent_coordinates.map(finite);
  }
  if (Array.isArray(node.data.hyperspherical_angles)) {
    return hypersphericalToCartesian(node.data.hyperspherical_angles);
  }
  return [];
}

function stableProjection(
  labels: string[],
  seed: string,
): number[][] {
  const rows = [[], [], []] as number[][];
  for (const label of labels) {
    const random = pseudoRandom(hashText(`${seed}|${label}`));
    const column = normalize([random(), random(), random()]);
    rows[0].push(column[0]);
    rows[1].push(column[1]);
    rows[2].push(column[2]);
  }
  return rows;
}

function fibonacciCaps(
  networks: string[],
): Record<string, [number, number, number]> {
  const caps: Record<string, [number, number, number]> = {};
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  networks.forEach((network, index) => {
    if (networks.length === 1) {
      caps[network] = [0, 0, 1];
      return;
    }
    const y = 1 - (2 * (index + 0.5)) / networks.length;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = index * goldenAngle;
    caps[network] = [
      Math.cos(angle) * radius,
      y,
      Math.sin(angle) * radius,
    ];
  });
  return caps;
}

function packIntoCap(
  direction: number[],
  center: [number, number, number],
  capAngle: number,
): [number, number, number] {
  const cosine = Math.max(-1, Math.min(1, dot(direction, center)));
  const sourceAngle = Math.acos(cosine);
  if (sourceAngle <= EPSILON) return [...center];
  const tangent = normalize(direction.map(
    (value, index) => value - cosine * center[index],
  ));
  const angle = capAngle * sourceAngle / Math.PI;
  return [
    Math.cos(angle) * center[0] + Math.sin(angle) * tangent[0],
    Math.cos(angle) * center[1] + Math.sin(angle) * tangent[1],
    Math.cos(angle) * center[2] + Math.sin(angle) * tangent[2],
  ];
}

function nodeFeatures(node: LatentRadarNode): {
  pressure: number;
  hydration: number;
  ionLoad: number;
  forwardCommitment: number;
  backwardCommitment: number;
} {
  const volume = Math.max(EPSILON, finite(node.data.volume));
  const solvent = Math.max(0, finite(node.data.solvent));
  const ions = Object.values(node.data.solubles ?? {}).reduce(
    (sum, value) => sum + Math.max(0, finite(value)),
    0,
  );
  return {
    pressure: Math.tanh(Math.log1p(Math.max(0, finite(node.data.pressure)))),
    hydration: Math.tanh(solvent / volume),
    ionLoad: Math.tanh(ions / volume),
    forwardCommitment: Math.tanh(
      Math.max(0, finite(node.data.forward_growth_interest)) / 3,
    ),
    backwardCommitment: Math.tanh(
      Math.max(0, finite(node.data.backward_growth_interest)) / 3,
    ),
  };
}

export class HypersphereProjector {
  private readonly seed: string;

  constructor(seed = "flux-radar-observation-v1") {
    this.seed = seed;
  }

  observe(
    nodes: LatentRadarNode[],
    width: number,
    height: number,
    intrinsicLabels: string[] = [],
  ): HypersphereObservationFrame {
    const networks = Array.from(new Set(nodes.map(networkKey))).sort();
    const intrinsicDimensionCount = nodes.reduce(
      (maximum, node) => Math.max(maximum, intrinsicCoordinates(node).length),
      0,
    );
    const hasMechanicalCoordinates = intrinsicDimensionCount > 0;
    const intrinsicDimensions = Array.from({ length: intrinsicDimensionCount }, (_, index) =>
      intrinsicLabels[index] ?? `intrinsic:${index}`
    );
    const dimensions = hasMechanicalCoordinates
      ? intrinsicDimensions
      : [
          ...SHARED_DIMENSIONS,
          ...networks.flatMap((network) => NETWORK_FEATURES.map(
            (feature) => `${network}:${feature}`,
          )),
        ];
    // Every dimension owns a deterministic random 3-vector. Adding a new
    // network therefore appends columns without rotating any existing one.
    const projection = stableProjection(dimensions, this.seed);
    const networkCaps = fibonacciCaps(networks);
    const capAngle = networks.length <= 1
      ? Math.PI * 0.92
      : Math.max(0.24, Math.min(0.68, 1.12 / Math.sqrt(networks.length)));
    const shellRadius = Math.max(40, Math.min(width, height) * 0.39);
    const centerX = width / 2;
    const centerY = height / 2;

    const observations = nodes.map((node): ProjectedObservationNode => {
      const network = networkKey(node);
      const features = nodeFeatures(node);
      const latent = new Array(dimensions.length).fill(0);
      if (hasMechanicalCoordinates) {
        intrinsicCoordinates(node).forEach((value, index) => {
          latent[index] = value;
        });
      } else {
        latent[0] = node.data.is_anchor || node.data.is_center ? 1 : 0;
        latent[1] = node.data.direction === "backward"
          ? -1
          : node.data.direction === "forward" ? 1 : 0;
        latent[2] = Math.tanh(finite(node.data.level ?? node.data.depth) / 4);
        latent[3] = features.pressure;
        latent[4] = features.hydration;
        latent[5] = features.ionLoad;
        latent[6] = Math.tanh(finite(node.data.local_evidence));
        latent[7] = Math.tanh(finite(node.data.path_mean));

        const networkOffset = SHARED_DIMENSIONS.length
          + networks.indexOf(network) * NETWORK_FEATURES.length;
        latent[networkOffset] = 1;
        latent[networkOffset + 1] = features.pressure;
        latent[networkOffset + 2] = features.hydration;
        latent[networkOffset + 3] = features.ionLoad;
        latent[networkOffset + 4] = features.forwardCommitment;
        latent[networkOffset + 5] = features.backwardCommitment;
      }

      const latentNorm = Math.sqrt(dot(latent, latent));
      const unit = normalize(latent);
      const projected = projection.map((row) => dot(row, unit)) as [
        number,
        number,
        number,
      ];
      const confidence = latentNorm <= EPSILON
        ? 0
        : Math.sqrt(dot(projected, projected));
      const cap = networkCaps[network] ?? [0, 0, 1];
      const shell = confidence <= EPSILON
        ? [...cap] as [number, number, number]
        : packIntoCap(normalize(projected), cap, capAngle);
      const perspective = 0.78 + 0.22 * ((shell[2] + 1) / 2);
      return {
        id: node.id,
        network,
        latent,
        unit,
        projected,
        shell,
        confidence: Math.min(1, confidence),
        screen: {
          x: centerX + shell[0] * shellRadius * perspective,
          y: centerY - shell[1] * shellRadius * perspective,
          depth: shell[2],
          scale: perspective,
        },
      };
    });

    return {
      version: 1,
      dimensions,
      projection,
      networkCaps,
      shellRadius,
      nodes: observations,
    };
  }
}
