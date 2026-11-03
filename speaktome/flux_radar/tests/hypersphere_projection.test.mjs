import assert from "node:assert/strict";
import {
  HypersphereProjector,
  NDForceAssembly,
  hypersphericalToCartesian,
} from "../webgl/webgl_renderer.js";

const close = (left, right, tolerance = 1e-9) => {
  assert.ok(Math.abs(left - right) <= tolerance, `${left} != ${right}`);
};
const norm = (vector) => Math.hypot(...vector);

const pole = hypersphericalToCartesian([Math.PI / 2, Math.PI / 2]);
close(pole[0], 0);
close(pole[1], 0);
close(pole[2], 1);

const projector = new HypersphereProjector("projection-test");
const main = {
  id: 1,
  data: {
    direction: "forward",
    level: 2,
    pressure: 1.25,
    volume: 2,
    solvent: 1.5,
    solubles: { sodium: 0.25 },
    local_evidence: 0.4,
    path_mean: 0.3,
    latent_coordinates: [0.2, -0.5, 0.7, 0.1],
  },
};
const cousin = {
  id: 2,
  data: {
    ...main.data,
    network_root: 42,
    direction: "backward",
    hyperspherical_angles: [Math.PI / 3, Math.PI / 4, Math.PI / 6],
    latent_coordinates: undefined,
  },
};

const first = projector.observe([main], 900, 560);
const repeated = projector.observe([main], 900, 560);
assert.deepEqual(first, repeated);
assert.equal(first.nodes[0].latent.length, first.dimensions.length);
close(norm(first.nodes[0].unit), 1);
close(norm(first.nodes[0].shell), 1);
assert.ok(Number.isFinite(first.nodes[0].screen.x));
assert.ok(first.nodes[0].confidence >= 0 && first.nodes[0].confidence <= 1);

const expanded = projector.observe([main, cousin], 900, 560);
assert.equal(Object.keys(expanded.networkCaps).length, 2);
for (const label of first.dimensions) {
  const before = first.dimensions.indexOf(label);
  const after = expanded.dimensions.indexOf(label);
  assert.ok(after >= 0, `missing stable dimension ${label}`);
  for (let row = 0; row < 3; row += 1) {
    close(first.projection[row][before], expanded.projection[row][after]);
  }
}
for (const node of expanded.nodes) close(norm(node.shell), 1);

const force = new NDForceAssembly(2, 4);
const forceNodes = [
  {
    id: 10,
    data: {
      is_anchor: true,
      level: 0,
      pressure: 1,
      volume: 1,
    },
  },
  {
    id: 11,
    data: {
      level: 1,
      direction: "forward",
      pressure: 1.5,
      volume: 1,
    },
  },
  {
    id: 20,
    data: {
      network_root: 20,
      is_center: true,
      level: 0,
      pressure: 1,
      volume: 1,
    },
  },
  {
    id: 21,
    data: {
      network_root: 20,
      level: -1,
      direction: "backward",
      pressure: 0.8,
      volume: 1,
    },
  },
];
force.sync(forceNodes, [
  { source: 10, target: 11 },
  { source: 20, target: 21 },
]);
const proof = force.relax(80);
assert.equal(proof.dimensions, 10);
assert.equal(proof.nodes, 4);
assert.equal(proof.edges, 2);
assert.ok(Number.isFinite(proof.residual));
assert.ok(Number.isFinite(proof.potentialEnergy));
assert.deepEqual(force.coordinates(10), new Array(10).fill(0));
assert.deepEqual(force.coordinates(20), new Array(10).fill(0));
for (const id of [11, 21]) {
  assert.equal(force.coordinates(id).length, 10);
  assert.ok(force.coordinates(id).every(Number.isFinite));
}
const forceSnapshot = force.snapshot();
assert.deepEqual(forceSnapshot.networks, ["main", "net:20"]);
assert.equal(Object.keys(force.habitatProximity()).length, 2);
const mechanicalObservation = projector.observe(
  forceNodes.map((node) => ({
    ...node,
    data: {
      ...node.data,
      latent_coordinates: force.coordinates(node.id),
    },
  })),
  900,
  560,
  forceSnapshot.dimensions,
);
assert.deepEqual(mechanicalObservation.dimensions, forceSnapshot.dimensions);
assert.equal(mechanicalObservation.nodes.find((node) => node.id === 10).confidence, 0);

console.log("hypersphere projection tests passed");
