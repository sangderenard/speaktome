export interface NDForceNode {
  id: number;
  data: {
    network_root?: number | null;
    is_anchor?: boolean;
    is_center?: boolean;
    level?: number | null;
    depth?: number;
    direction?: string | null;
    pressure?: number;
    volume?: number;
  };
}

export interface NDForceLink {
  source: number | NDForceNode;
  target: number | NDForceNode;
}

export interface NDRelaxationProof {
  dimensions: number;
  nodes: number;
  edges: number;
  substeps: number;
  residual: number;
  kineticEnergy: number;
  potentialEnergy: number;
}

export interface NDForceSnapshot {
  dimensions: string[];
  networks: string[];
  positions: Record<string, number[]>;
  proof: NDRelaxationProof | null;
}

const EPSILON = 1e-7;

function finite(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function endpointId(endpoint: number | NDForceNode): number {
  return typeof endpoint === "number" ? endpoint : endpoint.id;
}

function networkKey(node: NDForceNode): string {
  return node.data.network_root == null
    ? "main"
    : `net:${node.data.network_root}`;
}

function hashUnit(id: number, salt: number): number {
  let value = Math.imul((id ^ salt) >>> 0, 0x45d9f3b);
  value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x45d9f3b);
  value ^= value >>> 16;
  return (value >>> 0) / 0xffffffff;
}

function neighborOffsets(dimensions: number): number[][] {
  let offsets: number[][] = [[]];
  for (let dimension = 0; dimension < dimensions; dimension += 1) {
    offsets = offsets.flatMap((prefix) => [-1, 0, 1].map(
      (offset) => [...prefix, offset],
    ));
  }
  return offsets;
}

export class NDForceAssembly {
  readonly sharedDimensions: number;
  readonly localDimensions: number;

  private networks: string[] = [];
  private dimensionLabels: string[] = [];
  private nodes: NDForceNode[] = [];
  private nodeIndex = new Map<number, number>();
  private networkIndices = new Int32Array();
  private fixed = new Uint8Array();
  private levels = new Float32Array();
  private pressures = new Float32Array();
  private radii = new Float32Array();
  private positions = new Float32Array();
  private velocities = new Float32Array();
  private sources = new Int32Array();
  private targets = new Int32Array();
  private lastProof: NDRelaxationProof | null = null;
  private readonly collisionOffsets: number[][];
  private pipeStiffness = 0.72;
  private pipeRestLength = 1.05;
  private pressureExpansion = 0.06;

  constructor(sharedDimensions = 2, localDimensions = 4) {
    this.sharedDimensions = Math.max(0, Math.floor(sharedDimensions));
    this.localDimensions = Math.max(2, Math.floor(localDimensions));
    this.collisionOffsets = neighborOffsets(this.localDimensions);
  }

  configure(parameters: {
    pipeStiffness?: number;
    pipeRestLength?: number;
    pressureExpansion?: number;
  }): void {
    if (parameters.pipeStiffness != null) {
      this.pipeStiffness = Math.max(0, finite(parameters.pipeStiffness));
    }
    if (parameters.pipeRestLength != null) {
      this.pipeRestLength = Math.max(EPSILON, finite(parameters.pipeRestLength));
    }
    if (parameters.pressureExpansion != null) {
      this.pressureExpansion = Math.max(
        0,
        finite(parameters.pressureExpansion),
      );
    }
  }

  private rebuildDimensionLabels(): void {
    this.dimensionLabels = [
      ...Array.from(
        { length: this.sharedDimensions },
        (_, index) => `world:${index}`,
      ),
      ...this.networks.flatMap((network) => Array.from(
        { length: this.localDimensions },
        (_, index) => `${network}:space:${index}`,
      )),
    ];
  }

  private localOffset(networkIndex: number): number {
    return this.sharedDimensions + networkIndex * this.localDimensions;
  }

  sync(nodes: NDForceNode[], links: NDForceLink[]): void {
    const oldLabels = this.dimensionLabels;
    const oldPositions = this.positions;
    const oldVelocities = this.velocities;
    const oldNodeIndex = this.nodeIndex;
    const oldDimensionCount = oldLabels.length;

    for (const node of nodes) {
      const network = networkKey(node);
      if (!this.networks.includes(network)) this.networks.push(network);
    }
    this.rebuildDimensionLabels();
    const dimensionCount = this.dimensionLabels.length;
    const labelRemap = this.dimensionLabels.map((label) => oldLabels.indexOf(label));

    this.nodes = [...nodes];
    this.nodeIndex = new Map(nodes.map((node, index) => [node.id, index]));
    this.networkIndices = new Int32Array(nodes.length);
    this.fixed = new Uint8Array(nodes.length);
    this.levels = new Float32Array(nodes.length);
    this.pressures = new Float32Array(nodes.length);
    this.radii = new Float32Array(nodes.length);
    this.positions = new Float32Array(nodes.length * dimensionCount);
    this.velocities = new Float32Array(nodes.length * dimensionCount);

    nodes.forEach((node, nodeIndex) => {
      const networkIndex = this.networks.indexOf(networkKey(node));
      this.networkIndices[nodeIndex] = networkIndex;
      this.fixed[nodeIndex] = node.data.is_anchor || node.data.is_center ? 1 : 0;
      this.levels[nodeIndex] = finite(node.data.level ?? node.data.depth);
      this.pressures[nodeIndex] = Math.max(0, finite(node.data.pressure));
      this.radii[nodeIndex] = 0.16
        + 0.08 * Math.sqrt(Math.max(0, finite(node.data.volume)));

      const previousNode = oldNodeIndex.get(node.id);
      if (previousNode != null && oldDimensionCount > 0) {
        for (let dimension = 0; dimension < dimensionCount; dimension += 1) {
          const oldDimension = labelRemap[dimension];
          if (oldDimension < 0) continue;
          this.positions[nodeIndex * dimensionCount + dimension]
            = oldPositions[previousNode * oldDimensionCount + oldDimension];
          this.velocities[nodeIndex * dimensionCount + dimension]
            = oldVelocities[previousNode * oldDimensionCount + oldDimension];
        }
        return;
      }

      if (this.fixed[nodeIndex]) return;
      const localOffset = this.localOffset(networkIndex);
      const targetRadius = Math.max(0.7, Math.abs(this.levels[nodeIndex]) * 1.05);
      let squaredLength = 0;
      for (let local = 0; local < this.localDimensions; local += 1) {
        const value = hashUnit(node.id, 0x9e37 + local * 0x85eb) * 2 - 1;
        this.positions[nodeIndex * dimensionCount + localOffset + local] = value;
        squaredLength += value * value;
      }
      const scale = targetRadius / Math.sqrt(Math.max(EPSILON, squaredLength));
      for (let local = 0; local < this.localDimensions; local += 1) {
        this.positions[nodeIndex * dimensionCount + localOffset + local] *= scale;
      }
      if (this.sharedDimensions > 0) {
        this.positions[nodeIndex * dimensionCount]
          = (hashUnit(node.id, 0x27d4) * 2 - 1) * 0.08;
      }
    });

    const edgePairs = links.flatMap((link) => {
      const source = this.nodeIndex.get(endpointId(link.source));
      const target = this.nodeIndex.get(endpointId(link.target));
      return source == null || target == null ? [] : [[source, target]];
    });
    this.sources = new Int32Array(edgePairs.map(([source]) => source));
    this.targets = new Int32Array(edgePairs.map(([, target]) => target));
  }

  private addPipeForces(forces: Float32Array): number {
    const dimensions = this.dimensionLabels.length;
    let energy = 0;
    for (let edge = 0; edge < this.sources.length; edge += 1) {
      const source = this.sources[edge];
      const target = this.targets[edge];
      const sourceNetwork = this.networkIndices[source];
      const targetNetwork = this.networkIndices[target];
      const activeDimensions: number[] = Array.from(
        { length: this.sharedDimensions },
        (_, index) => index,
      );
      // A pipe inside one network acts in that network's local subspace.
      // A bridge between networks acts only in explicitly shared world
      // dimensions; it never drags either endpoint through foreign axes.
      if (targetNetwork === sourceNetwork) {
        for (let local = 0; local < this.localDimensions; local += 1) {
          activeDimensions.push(this.localOffset(sourceNetwork) + local);
        }
      }

      let distanceSquared = 0;
      for (const dimension of activeDimensions) {
        const delta = this.positions[target * dimensions + dimension]
          - this.positions[source * dimensions + dimension];
        distanceSquared += delta * delta;
      }
      const distance = Math.sqrt(Math.max(EPSILON, distanceSquared));
      const pressureExpansion = this.pressureExpansion * Math.log1p(
        (this.pressures[source] + this.pressures[target]) * 0.5,
      );
      const restLength = this.pipeRestLength + pressureExpansion;
      const extension = distance - restLength;
      const stiffness = this.pipeStiffness;
      const magnitude = stiffness * extension / distance;
      energy += 0.5 * stiffness * extension * extension;
      for (const dimension of activeDimensions) {
        const delta = this.positions[target * dimensions + dimension]
          - this.positions[source * dimensions + dimension];
        const force = magnitude * delta;
        forces[source * dimensions + dimension] += force;
        forces[target * dimensions + dimension] -= force;
      }
    }
    return energy;
  }

  private addLocalShellForces(forces: Float32Array): number {
    const dimensions = this.dimensionLabels.length;
    let energy = 0;
    for (let node = 0; node < this.nodes.length; node += 1) {
      if (this.fixed[node]) continue;
      const offset = this.localOffset(this.networkIndices[node]);
      let radiusSquared = 0;
      for (let local = 0; local < this.localDimensions; local += 1) {
        const value = this.positions[node * dimensions + offset + local];
        radiusSquared += value * value;
      }
      const radius = Math.sqrt(Math.max(EPSILON, radiusSquared));
      const targetRadius = Math.max(0.7, Math.abs(this.levels[node]) * 1.05);
      const displacement = radius - targetRadius;
      const stiffness = 0.24;
      energy += 0.5 * stiffness * displacement * displacement;
      for (let local = 0; local < this.localDimensions; local += 1) {
        const index = node * dimensions + offset + local;
        forces[index] -= stiffness * displacement * this.positions[index] / radius;
      }
      for (let shared = 0; shared < this.sharedDimensions; shared += 1) {
        const index = node * dimensions + shared;
        forces[index] -= 0.025 * this.positions[index];
      }
    }
    return energy;
  }

  private addCollisionForces(forces: Float32Array): number {
    const dimensions = this.dimensionLabels.length;
    const cellSize = 0.48;
    const grids = this.networks.map(() => new Map<string, number[]>());
    let energy = 0;
    for (let node = 0; node < this.nodes.length; node += 1) {
      const network = this.networkIndices[node];
      const offset = this.localOffset(network);
      const cell = Array.from({ length: this.localDimensions }, (_, local) =>
        Math.floor(this.positions[node * dimensions + offset + local] / cellSize)
      );
      const grid = grids[network];
      for (const neighborOffset of this.collisionOffsets) {
        const key = cell.map(
          (coordinate, dimension) => coordinate + neighborOffset[dimension],
        ).join(",");
        for (const other of grid.get(key) ?? []) {
          let distanceSquared = 0;
          for (let local = 0; local < this.localDimensions; local += 1) {
            const dimension = offset + local;
            const delta = this.positions[node * dimensions + dimension]
              - this.positions[other * dimensions + dimension];
            distanceSquared += delta * delta;
          }
          const distance = Math.sqrt(Math.max(EPSILON, distanceSquared));
          const minimum = this.radii[node] + this.radii[other] + 0.08;
          if (distance >= minimum) continue;
          const overlap = minimum - distance;
          const stiffness = 0.9;
          energy += 0.5 * stiffness * overlap * overlap;
          for (let local = 0; local < this.localDimensions; local += 1) {
            const dimension = offset + local;
            const delta = this.positions[node * dimensions + dimension]
              - this.positions[other * dimensions + dimension];
            const force = stiffness * overlap * delta / distance;
            forces[node * dimensions + dimension] += force;
            forces[other * dimensions + dimension] -= force;
          }
        }
      }
      const ownKey = cell.join(",");
      const occupants = grid.get(ownKey) ?? [];
      occupants.push(node);
      grid.set(ownKey, occupants);
    }
    return energy;
  }

  relax(substeps = 32, timeStep = 0.035): NDRelaxationProof {
    const dimensions = this.dimensionLabels.length;
    const forces = new Float32Array(this.positions.length);
    let potentialEnergy = 0;
    let residual = 0;
    for (let step = 0; step < substeps; step += 1) {
      forces.fill(0);
      potentialEnergy = this.addPipeForces(forces)
        + this.addLocalShellForces(forces)
        + this.addCollisionForces(forces);
      let squaredResidual = 0;
      const damping = Math.exp(-3.8 * timeStep);
      for (let node = 0; node < this.nodes.length; node += 1) {
        for (let dimension = 0; dimension < dimensions; dimension += 1) {
          const index = node * dimensions + dimension;
          if (this.fixed[node]) {
            this.positions[index] = 0;
            this.velocities[index] = 0;
            continue;
          }
          squaredResidual += forces[index] * forces[index];
          this.velocities[index] = (
            this.velocities[index] + forces[index] * timeStep
          ) * damping;
          this.velocities[index] = Math.max(
            -5,
            Math.min(5, this.velocities[index]),
          );
          this.positions[index] += this.velocities[index] * timeStep;
        }
      }
      residual = Math.sqrt(
        squaredResidual / Math.max(1, this.nodes.length * dimensions),
      );
    }
    let kineticEnergy = 0;
    for (const velocity of this.velocities) {
      kineticEnergy += 0.5 * velocity * velocity;
    }
    this.lastProof = {
      dimensions,
      nodes: this.nodes.length,
      edges: this.sources.length,
      substeps,
      residual,
      kineticEnergy,
      potentialEnergy,
    };
    return this.lastProof;
  }

  coordinates(nodeId: number): number[] {
    const node = this.nodeIndex.get(nodeId);
    if (node == null) return [];
    const dimensions = this.dimensionLabels.length;
    return Array.from(
      this.positions.subarray(node * dimensions, (node + 1) * dimensions),
    );
  }

  habitatProximity(): Record<string, number> {
    const dimensions = this.dimensionLabels.length;
    const proximity: Record<string, number> = {};
    for (let node = 0; node < this.nodes.length; node += 1) {
      if (this.fixed[node]) continue;
      const offset = this.localOffset(this.networkIndices[node]);
      let radiusSquared = 0;
      for (let local = 0; local < this.localDimensions; local += 1) {
        const value = this.positions[node * dimensions + offset + local];
        radiusSquared += value * value;
      }
      const radius = Math.sqrt(radiusSquared);
      const targetRadius = Math.max(0.7, Math.abs(this.levels[node]) * 1.05);
      proximity[String(this.nodes[node].id)] = Math.max(
        0,
        1 - Math.abs(radius - targetRadius) / 1.05,
      );
    }
    return proximity;
  }

  snapshot(): NDForceSnapshot {
    return {
      dimensions: [...this.dimensionLabels],
      networks: [...this.networks],
      positions: Object.fromEntries(
        this.nodes.map((node) => [String(node.id), this.coordinates(node.id)]),
      ),
      proof: this.lastProof,
    };
  }
}
