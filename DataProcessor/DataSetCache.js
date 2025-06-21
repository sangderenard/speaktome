class DataSetCache {
  constructor() {
    this.datasets = {};
    this.extrema = {
      domain: { min: Infinity, max: -Infinity },
      range: {},
    };
    this.threadTimers = {}; // Object to store timer IDs for each dataset
  }

  initCache(name, dataset) {
    // Add the dataset to the cache with its name
    this.datasets[name] = dataset;

    // Update min and max domain coordinates of cache
    const { min, max } = dataset.getDomainExtrema();
    this.extrema.domain.min = Math.min(this.extrema.domain.min, min);
    this.extrema.domain.max = Math.max(this.extrema.domain.max, max);

    // Update range axes
    for (const axis in dataset.rangeAxes) {
      if (!(axis in this.extrema.range)) {
        this.extrema.range[axis] = {
          min: Infinity,
          max: -Infinity,
        };
      }
      const { min, max } = dataset.getRangeExtrema(axis);
      this.extrema.range[axis].min = Math.min(this.extrema.range[axis].min, min);
      this.extrema.range[axis].max = Math.max(this.extrema.range[axis].max, max);
    }

    // Launch thread for curating the dataset
    this.launchDatasetThread(name);
  }

  launchDatasetThread(name) {
    const dataset = this.datasets[name];
    const thread = () => {
      // TODO: Densify the dataset and update the cache with new values
      // from control points
      // Lock dataset before modifying it
      dataset.lock();
      // TODO: Implement densification and updating of cache using control points
      // Unlock dataset after modifying it
      dataset.unlock();

      // Call thread again after some time
      const timerId = setTimeout(thread, dataset.maxSize / 1000);
      this.threadTimers[name] = timerId;
    };

    // Call thread for the first time
    const timerId = setTimeout(thread, dataset.maxSize / 1000);
    this.threadTimers[name] = timerId;
  }

  stopDatasetThread(name) {
    const timerId = this.threadTimers[name];
    if (timerId) {
      clearTimeout(timerId);
      delete this.threadTimers[name];
    }
  }

  getPoint(point, datasetName) {
    // Find the dataset with the given name
    const dataset = this.datasets[datasetName];
    if (!dataset) {
      return null;
    }

    // Find the resolution that is appropriate for the point
    const resolution = this.findResolutionForPoint(point);

    // Find the nearest point in the cache at the given resolution
    const nearestPoint = dataset.findNearestPoint(point, resolution);

    // If the nearest point is within a certain distance from the requested point,
    // return the value of the nearest point
    if (nearestPoint.distanceTo(point) < this.pointThreshold) {
      return nearestPoint.value;
    }

    // Interpolate the value of the requested point
    const interpolatedValue = dataset.interpolatePoint(point, resolution);

    return interpolatedValue;
  }

  setPoint(point, value) {
    // Set the value of a specific point
    // TODO
  }

  densifyCache(resolution) {
    // Increase the number of points in the cache for a given resolution
    // TODO
  }