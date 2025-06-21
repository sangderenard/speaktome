class RelativeCoordinates {
  constructor(latSemicircles, lonSemicircles) {
    this.latDegrees = this.semicirclesToDegrees(latSemicircles);
    this.lonDegrees = this.semicirclesToDegrees(lonSemicircles);
  }

  semicirclesToDegrees(semicircles) {
    return semicircles * (180 / Math.pow(2, 31));
  }

  degreesToRadians(degrees) {
    return degrees * (Math.PI / 180);
  }

  haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371000; // Earth's radius in meters
    const dLat = this.degreesToRadians(lat2 - lat1);
    const dLon = this.degreesToRadians(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.degreesToRadians(lat1)) *
        Math.cos(this.degreesToRadians(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  getRelativeCoordinates(latSemicircles, lonSemicircles) {
    const latDegrees = this.semicirclesToDegrees(latSemicircles);
    const lonDegrees = this.semicirclesToDegrees(lonSemicircles);

    const y = this.haversineDistance(this.latDegrees, this.lonDegrees, latDegrees, this.lonDegrees);
    const x = this.haversineDistance(this.latDegrees, this.lonDegrees, this.latDegrees, lonDegrees);

    const eastToWest = this.lonDegrees > lonDegrees ? -x : x;
    const southToNorth = this.latDegrees > latDegrees ? -y : y;

    return { x: eastToWest, y: southToNorth };
  }
}

class Point {
		constructor(coordinates, point = null) {
			const initializeAxes = () => {
				  this.coordinates = {};
				  standardAxes.forEach(axis => {
					  if(point && point.coordinates && point.coordinates[axis] != null){
						this.coordinates[axis] = point.coordinates[axis];  
					  }else{
				    	this.coordinates[axis] = null;
				    }
				  });
				}
			initializeAxes();
			
			if (!Array.isArray(coordinates)) {
				  for (const key in coordinates) {
				    if (!this.coordinates) {
				      this.coordinates = { [key]: coordinates[key] };
				    } else if (this.coordinates.hasOwnProperty(key)) {
				      this.coordinates[key] = coordinates[key];
				    } else {
				      this.coordinates = { ...this.coordinates, [key]: coordinates[key] };
				      //////////////////////////////////console.log(this.coordinates);
				    }
				  }
				} else if(coordinates){
				  this.coordinates = { x: coordinates[0], y: coordinates[1] };
				}
		}
		  roundCoordinates(sigfig) {
    const rounded = {};
    for (const [key, value] of Object.entries(this.coordinates)) {
      if (typeof value === 'number') {
        rounded[key] = parseFloat(value.toFixed(sigfig));
      } else {
        rounded[key] = value;
      }
    }
    return new Point(rounded);
  }
		  distanceTo(otherPoint, axes) {
			    // Calculate Euclidean distance between two points in the given set of axes
			    const squaredDistances = [];
			    axes.forEach((axis) => {
			      const delta = this.coordinates[axis] - otherPoint.coordinates[axis];
			      squaredDistances.push(delta * delta);
			    });
			    const sumOfSquares = squaredDistances.reduce((sum, distance) => sum + distance, 0);
			    return Math.sqrt(sumOfSquares);
			  }
			  static calculateAxesStats(points) {
			    const allAxes = new Set();
			    const axisValues = {};
			    const axisStats = {};
//////console.log("CALCULATING AXES STATS FOR ", points);
			    // Loop over all points and axes, and collect axis values
			    points.forEach((point, index) => {
			      Object.keys(point.coordinates).forEach((axis) => {
			        if(point.coordinates[axis] != null){
			        allAxes.add(axis);
			        const value = point.coordinates[axis];
			        if (axisValues[axis]) {
			          axisValues[axis].push({ index, value });
			        } else {
			          axisValues[axis] = [{ index, value }];
			        }
			        }
			      });
			    });

			    // Loop over each axis and calculate its statistics
			    allAxes.forEach((axis) => {
			      const values = axisValues[axis].map(({ value }) => value);
			      //////////////////////console.warn(values);
			      const min = Math.min(...values);
			      const max = Math.max(...values);
			      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
			      const sortedValues = values.sort((a, b) => a - b);
			      const middle = Math.floor(sortedValues.length / 2);
			      const isEven = sortedValues.length % 2 === 0;
			      const median = isEven ? (sortedValues[middle - 1] + sortedValues[middle]) / 2 : sortedValues[middle];
			      const valueCounts = {};
			      values.forEach((value) => {
			        valueCounts[value] = (valueCounts[value] || 0) + 1;
			      });
			      const mode = Object.keys(valueCounts).reduce((a, b) => valueCounts[a] > valueCounts[b] ? a : b);
			      const rms = Math.sqrt(values.reduce((sum, value) => sum + value ** 2, 0) / values.length);

			      // Store the axis statistics in an object keyed by the axis name
			      axisStats[axis] = { values, min, max, mean, median, mode, rms };
			    });

			    // Loop over all points and add the index of any axis values that are missing
			    points.forEach((point) => {
			      Object.keys(point.coordinates).forEach((axis) => {
			        if (axis === 'R' || axis === 'G' || axis === 'B' || axis === 'A') {
			          return;
			        }
			        if (!axisStats[axis]) {
			          return;
			        }
			        const index = axisValues[axis].find(({ value }) => value === point.coordinates[axis])?.index;
			        if (index !== undefined) {
			          if (!axisStats[axis].indices) {
			            axisStats[axis].indices = [];
			          }
			          axisStats[axis].indices.push(index);
			        }
			      });
			    });

			    // Return the axis statistics object
			    //////////////////////////////////console.log("FOUND THIS: ",axisStats);
			    return axisStats;
			  }
			
		 static sortPoints(points) {
			    points.sort((a, b) => a.coordinates.x - b.coordinates.x);
			    const yValues = points.map(point => point.coordinates.y);
			    for (let i = 0; i < points.length; i++) {
			      points[i].coordinates.y = yValues[i];
			    }
			    return points;
			  }
			  
			  static parsePointString = (pointString) => {
  const points = [];
  const pointsData = pointString.split('|');

  for (let i = 0; i < pointsData.length; i++) {
    const pointData = pointsData[i].split(',');
    const coordinates = {};

    for (let j = 0; j < pointData.length; j++) {
      const [label, value] = pointData[j].split(':');
      coordinates[label] = value;
    }

    const point = new Point(coordinates);
    if (Point.validPoint(point)) {
      points.push(point);
    }else{
		return dataset;
	}

  }


  return points;
};
clone(){
	return new Point(this.coordinates);
}		 
  static validPoint(point) {
    return point && point.coordinates && Object.keys(point.coordinates).length > 0;
  }	
		}
class PointCollection {
  constructor(points, progressOptions, delay = 0) {
    this.points = points;
    this.axisStats = {};
    this.progressOptions = {};

    if(progressOptions && progressOptions.chainlink && progressOptions.id){
	
    this.progressOptions.id = progressOptions.chainlink.inputControl.progressBarManager.getID()+":"+progressOptions.id;
     this.progressOptions.chainlink = progressOptions.chainlink;
     this.progressOptions.progress = (progress) => {
		 this.progressOptions.chainlink.inputControl.progress(this.progressOptions.id, progress, "Point Collection", null);
	 }

    }else{
		this.progressOptions = null;
	} 
    this.calculationPromise = this.calculateDataAsync(this.progressOptions, delay);
  }

 async calculateDataAsync(progressOptions = this.progressOptions, delay) {
    // Create a MyWorker instance with the progressOptions.chainlink
    const worker = new MyWorker(
      progressOptions ? progressOptions.chainlink : null,
      async (i) => {
        const axis = PointCollection.getAllAxes(points)[i];
        const axisValues = PointCollection.getAxisValues(points, axis);
        return await PointCollection.calculateAxisStats(axis, axisValues, delay);
      },
      'Calculating axes stats',
      PointCollection.getAllAxes(this.points).length,
      delay
    );

    // Use the runThreaded method to calculate the axis stats asynchronously
    const resolvedStats = await worker.runThreaded({ 'points':this.points}, ['PointCollection', 'Point', 'MyWorker']);
////console.warn(resolvedStats);
    // Assign the resolved stats to the axisStats object
    resolvedStats.forEach(axisStat => {
      this.axisStats[axisStat.axis] = axisStat;
    });

    this.calculationPromise = null;
  }

  getPointMetadata(point) {
    const metadata = {};

    Object.keys(point.coordinates).forEach((axis) => {
      if (axis === 'R' || axis === 'G' || axis === 'B' || axis === 'A') {
        return;
      }
      if (!this.axisStats[axis]) {
        return;
      }
      metadata[axis] = {
        value: point.coordinates[axis],
        min: this.axisStats[axis].min,
        max: this.axisStats[axis].max,
        mean: this.axisStats[axis].mean,
        median: this.axisStats[axis].median,
        mode: this.axisStats[axis].mode,
        rms: this.axisStats[axis].rms
      };
    });

    return metadata;
  }

  static getAllAxes(points) {
    const allAxes = new Set();
    points.forEach((point) => {
      Object.keys(point.coordinates).forEach((axis) => {
        if (point.coordinates[axis] != null) {
          allAxes.add(axis);
        }
      });
    });
    return Array.from(allAxes);
  }

  static getAxisValues(points, axis) {
    const axisValues = [];
    points.forEach((point, index) => {
      const value = point.coordinates[axis];
      if (value != null) {
        axisValues.push({ index, value });
      }
    });
    return axisValues;
  }

  static async calculateAxisStats(axis, axisValues, delay) {
    const worker = new MyWorker(null, async (i) => {
      const valueObj = axisValues[i];
      return { index: valueObj.index, value: valueObj.value };
    }, 'Calculating axis stats', axisValues.length, delay);
    const valuesWithIndices = await worker.run();

    const values = valuesWithIndices.map(({ value }) => value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const sortedValues = values.sort((a, b) => a - b);
    const middle = Math.floor(sortedValues.length / 2);
    const isEven = sortedValues.length % 2 === 0;
    const median = isEven ? (sortedValues[middle - 1] + sortedValues[middle]) / 2 : sortedValues[middle];
    const valueCounts = {};
    values.forEach((value) => {
      valueCounts[value] = (valueCounts[value] || 0) + 1;
    });
    const mode = Object.keys(valueCounts).reduce((a, b) => valueCounts[a] > valueCounts[b] ? a : b);
    const rms = Math.sqrt(values.reduce((sum, value) => sum + value ** 2, 0) / values.length);

    return { axis, values, min, max, mean, median, mode, rms };
  }

static async calculateAxesStats(points, progressOptions, delay = 0) {
  const allAxes = PointCollection.getAllAxes(points);
  const axisStats = {};

  // Create an array to hold all the promises
  const promises = [];

  for (const axis of allAxes) {
    const axisValues = PointCollection.getAxisValues(points, axis);
    // Call calculateAxisStats without await and store the promise in the promises array
    const axisStatPromise = PointCollection.calculateAxisStats(axis, axisValues, delay);
    promises.push(axisStatPromise);
  }

  // Wait for all promises to resolve
  const resolvedStats = await Promise.all(promises);

  // Assign the resolved stats to the axisStats object
  resolvedStats.forEach(axisStat => {
    axisStats[axisStat.axis] = axisStat;
  });

  return axisStats;
}
}

class Interval {
	  constructor(start, end = null, point = null) {
		  ////////////////////////////////////////console.log("Adding interval..." + start +":"+end+":"+JSON.stringify(point));
		    if (start === null) {
		      if (!end || !point) {
		        throw new Error('Missing required arguments');
		      }
		      const [startTime, power] = [point.coordinates.x, point.coordinates.y];
		    //  //////////////////////////////////////console.log("power at point: " + power);
		      this.start = startTime === undefined ? null : startTime;
		      this.power = power === undefined ? null : power;
		      this.end = end.coordinates.x;
		      this.duration = this.end - this.start;
		    } else {
		      if (!point) {
		        throw new Error('Missing required arguments');
		      }
		      ////////////////////////////////////////console.log("adding interval via start, duration, power");
		     // //////////////////////////////////////console.log(JSON.stringify(point));

		      //////////////////////////////////////console.trace();
		      const [duration, power] = [point.coordinates.x, point.coordinates.y];
		      ////////////////////////////////////////console.log("duration and power: " + duration + ":" + power);
		      this.start = start === undefined ? null : start;
		      this.power = power === undefined ? null : power;
		      this.duration = duration === undefined ? null : duration;
		      this.end = this.start + this.duration;
		    }
		    ////////////////////////////////////////console.log("added interval: " + JSON.stringify(this));
		    if(this.duration){
		    this.refresh();
	  }
		  }

	  getStart() {
	    return this.start;
	  }

	  getEnd() {
	    return this.end;
	  }

	  getDuration() {
	    return this.duration;
	  }

	  getPower() {
	    return this.power;
	  }

	  setStart(start) {
	    this.start = start;
	    this.refresh();
	  }

	  setEnd(end) {
	    this.end = end;
	    this.refresh();
	  }

	  setDuration(duration) {
	    this.duration = duration;
	    this.refresh();
	  }

	  setPower(power) {
	    this.power = power;
	  }

	  refresh(signed = true) {
		  if (isNaN(this.start) || isNaN(this.end) || isNaN(this.duration) || isNaN(this.power)) {
		    ////////////////////////////////////console.warn(`Invalid value for interval with start=${this.start}, end=${this.end}, duration=${this.duration}, power=${this.power}`);
		  }

		  if (this.start < 0 || this.end < 0 || this.duration <= 0) {
		    //////////////////////////////////////console.warn(`Invalid value for interval with start=${this.start}, end=${this.end}, duration=${this.duration}, power=${this.power}`);
		  }

		  if (!signed && this.power < 0) {
		    ////////////////////////////////////console.warn(`Negative power value for interval with start=${this.start}, end=${this.end}, duration=${this.duration}, power=${this.power}`);
		  }

		  const endTime = this.start + this.duration;
		  if (this.start >= this.end || this.start >= endTime || this.end <= this.start || this.end > endTime) {
		    ////////////////////////////////////console.warn(`Invalid start and end times for interval with start=${this.start}, end=${this.end}, duration=${this.duration}, power=${this.power}`);
		  }
		  this.end = endTime;
		}
	}
	class WorkoutPlan{
		constructor(intervals){
			for(let i = 0; i < intervals.length; i++){
				//construct array of these {duration: x, power: y}
				//this.push maybe?
			}
		}
	}

class DataSet {
		  constructor(points, intervals, chainlink = null) {
			  this.sigfig = 3;
			  this.inheritAxesData = this.inheritAxesData.bind(this);
			  //this.clone = this.clone.bind(this);
			  this.replacePoints = this.replacePoints.bind(this);
			  
		    	this.powerThreshold = .1;
		    	this.timeThreshold = .01;
		    	this.minInterval = .01;
			  if (intervals && !points) {

			      this.points = [];
			      this.intervals = intervals;
			      intervals.forEach((interval) => {
			        const point = interval.data;
			        this.points.push(point);
			      });
			    } else if (points && !intervals) {
  this.points = points.map((p) => {
    const { coordinates } = p;
    const newCoordinates = {};

    for (const key in coordinates) {
      if (coordinates.hasOwnProperty(key)) {
        newCoordinates[key] = coordinates[key];
      }
    }

    return { ...p, coordinates: newCoordinates };
  });
  ////////////console.warn(this.points);

  // Get the first two axes in the coordinates object of the first point
  if(!points[0]){
	  return;
  }
  const firstPoint = points[0];
  const axes = Object.keys(firstPoint.coordinates);
  const firstAxis = axes[0];
  const secondAxis = axes.length > 1 ? axes[1] : null;
try{
  this.intervals = points.map((point) => {
    const { coordinates } = point;
    const start = coordinates[firstAxis];
    const end = start;
    const data = secondAxis ? coordinates[secondAxis] : 0;

    return new Interval( start, end, new Point({x:end-start, y: data}) );
  });
  }catch{
	  console.warn("can't create intervals");
  }

			    //  const intervalEnds = new Set(this.intervals.map((i) => i.end));
			      //if (intervalEnds.size !== this.intervals.length) {
			        ////////////////////////////////////console.error("Points not sorted or contain duplicate x values");
			        //return null;
			      //}
			    } else if (points && intervals) {
			    //	//////////////////////////////////////console.log(points);
			      this.points = points;
			      this.intervals = intervals;
			    } else if (intervals === undefined && points === undefined) {
			      this.points = [];
			      this.intervals = [];
			    } else {
			      ////////////////////////////////////console.warn("No intervals or points provided");
			      this.points = [];
			      this.intervals = [];
			      return null;
			    }
if (this.points && this.points.length) {
  // First sort by the "i" value (if available) and then by index
  this.points.sort((a, b) => {
    if (a.coordinates.i !== undefined && b.coordinates.i !== undefined) {
      return a.coordinates.i - b.coordinates.i;
    } else if (a.coordinates.i !== undefined) {
      return -1;
    } else if (b.coordinates.i !== undefined) {
      return 1;
    } else {
      return this.points.indexOf(a) - this.points.indexOf(b);
    }
  });
 
  if (points.length > 2 && Object.keys(points[0]).length > 1) {
    ////////////////////console.error(this);
    this.setInterpolator();
  }
}
			  }
			  
static defaultColor = (() => { 
	let counter = 0;
	
	return (channel)=> {
	counter++;
  const epoch = Math.floor(Date.now() ); // Get current epoch in seconds
  const flashPeriod = 0.333; // The flash period, 0.333 seconds for three times a second

  // Determine the base color according to the epoch and flash period
  const baseColor = ((epoch + counter )% flashPeriod) < (flashPeriod / 2) ? 33 : 255;

  let defaultValue;
  if (channel === 'A') {
    defaultValue = 127;
  } else if (channel === 'R' || channel === 'G') {
    defaultValue = baseColor * 0.6 + Math.random() * 0.4 * baseColor;
  } else if (channel === 'B') {
    defaultValue = baseColor === 255 ? 0 : baseColor * 0.6 + Math.random() * 0.4 * baseColor;
  }

  return defaultValue;
  };
})();
getColorFuncs = function(domain = 'index', interpolationType = 'linear') {
  // Create a single interpolator for the domain axis
  const domainValues = this.points.map((point, index) => domain === 'index' ? index : point.coordinates[domain]);
  const domainInterpolator = new Interpolator(domainValues, [domainValues], interpolationType);

  // Create a separate interpolator for each color channel
  const colorInterpolators = {};
  ['R', 'G', 'B', 'A'].forEach(channel => {
    const channelValues = this.points.map(point => point.coordinates[channel] !== undefined ? point.coordinates[channel] : null);
const interpolator = new Interpolator(domainValues, [channelValues.map(val => val !== null ? val : DataSet.defaultColor(channel))], interpolationType);
colorInterpolators[channel] = interpolator.evaluate.bind(interpolator);
  });

  // Return an object with a function for each color channel that takes a domain value and returns the interpolated color value for that channel
  const colorFuncs = {};
  ['R', 'G', 'B', 'A'].forEach(channel => {
    colorFuncs[channel] = (domainValue) => {
      const channelInterpolator = colorInterpolators[channel];
      const interpolatedValue = channelInterpolator(domainValue);
      return interpolatedValue !== undefined ? interpolatedValue : DataSet.defaultColor(channel);
    };
  });

  colorFuncs[domain] = (domainValue) => {
    const interpolatedDomainValue = domainInterpolator.evaluate(domainValue);
    return interpolatedDomainValue;
  };

  return colorFuncs;
};
			  
	async inheritAxesData(points, progressOptions = null, interpType = 'linear', method = {mappingType: 'simple', dataInheritance: 'simple', detailInheritance: 'simple'}, range = ['t','u','x', 'theta'], domain = ['y', 'r'], detail = ['R', 'G', 'B', 'A']){
		 if(!this?.points[0]){
			 return points;
		 }
		 if(progressOptions){
			 progressOptions.id = progressOptions.chainlink?.inputControl?.progressBarManager?.getID()+":"+progressOptions.id;
		 }
		 return this.fillChildAxes(await this.processInheritedData(await this.buildAxesData(points, progressOptions, interpType, method, range, domain, detail)));
	 }
	 getRange(axis) {
  const values = this.points.map(point => point.coordinates[axis]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { min, max };
}
	async buildAxesData(points, progressOptions, interpType, method, range, domain, detail) {
		  if (this.points.length < 1) {
		    return points;
		  }

  const parentPointCollection = new PointCollection(this.points, progressOptions);
  const childPointCollection = new PointCollection(points, progressOptions);

  await Promise.all([
    parentPointCollection.calculationPromise,
    childPointCollection.calculationPromise,
  ]);

  const parentAxesStats = parentPointCollection.axisStats;
  const childAxesStats = childPointCollection.axisStats;


	//	  const parentAxesStats = Point.calculateAxesStats(this.points);
		//  const childAxesStats = Point.calculateAxesStats(points);

		  const fullAxes = Array.from(new Set([
			  ...Object.keys(parentAxesStats).filter(axis => parentAxesStats[axis].values.every(val => val != null)),
			  ...Object.keys(childAxesStats).filter(axis => childAxesStats[axis].values.every(val => val != null))
			]));

		  domain = domain.filter(axis => fullAxes.includes(axis));
		  range = range.filter(axis => fullAxes.includes(axis));

		  const excludedAxes = [...new Set([...domain, ...range].filter(axis => !fullAxes.includes(axis)))];
		  detail = [...new Set([...detail, ...excludedAxes])];
		  
		  let mainDomain = null;
		  let domainFound = false;

		  // Search the domain axes
		  for (let i = 0; i < domain.length && !domainFound; i++) {
		    const axis = domain[i];
		    if (parentAxesStats[axis] && childAxesStats[axis]) {
		      const parentAscending = parentAxesStats[axis].values.every((val, i, arr) => i === 0 || val > arr[i - 1]);
		      const childAscending = childAxesStats[axis].values.every((val, i, arr) => i === 0 || val > arr[i - 1]);
		      if (parentAscending === childAscending && !parentAxesStats[axis].values.every((val, i) => val === childAxesStats[axis].values[i])) {
		        mainDomain = axis;
		        domainFound = true;
		      }
		    }
		  }

		  // Search the range axes if no domain axes were found
		  if (!domainFound) {
		    for (let i = 0; i < range.length && !domainFound; i++) {
		      const axis = range[i];
		      if (parentAxesStats[axis] && childAxesStats[axis]) {
		        const parentAscending = parentAxesStats[axis].values.every((val, i, arr) => i === 0 || val > arr[i - 1]);
		        const childAscending = childAxesStats[axis].values.every((val, i, arr) => i === 0 || val > arr[i - 1]);
		        if (parentAscending === childAscending && !parentAxesStats[axis].values.every((val, i) => val === childAxesStats[axis].values[i])) {
		          mainDomain = axis;
		          domainFound = true;
		        }
		      }
		    }
		  }

		  // If no suitable domain axis was found, select the indices as the domain
		  if (!domainFound) {
		    mainDomain = 'index';
	    
		  }
		  let rangeAxes = [];
		  
		  const rangeAxesToCheck = [...range, ...domain];

		  // Check for suitable range axes
		  rangeAxesToCheck.forEach((axis) => {
		    if (axis !== mainDomain && parentAxesStats[axis] && childAxesStats[axis]) {
		      rangeAxes.push(axis);
		    }
		  });

		  // If no suitable range axes are found, select them from the unused shared axes
		  if (rangeAxes.length === 0) {
		    const unusedAxes = fullAxes.filter(axis => !rangeAxesToCheck.includes(axis));
		    unusedAxes.forEach(axis => {
		      if (childAxesStats[axis]) {
		        rangeAxes.push(axis);
		      }
		    });
		  }

		  let domainValues = [];
		  let rangeValues = {};
		  let colorValues = {};
		  const childIndices = Array.from(Array(points.length).keys());
		const parentIndices = Array.from(Array(this.points.length).keys());	
		
		//////////////////////console.warn("INDICES: ", parentIndices, childIndices);
		
	const indices = {parent: parentIndices, child:childIndices};

		  points.forEach((point, index) => {
		    let domainValue;
		    if (mainDomain === 'index') {
		      domainValue = index;
		    } else {
		      domainValue = point.coordinates[mainDomain];
		    }
		    if (domainValue !== undefined && !isNaN(domainValue)) {
		      domainValues.push(domainValue);
		    }
		    rangeAxes.forEach((axis) => {
		      if (!rangeValues[axis] && childAxesStats[axis]) {
		        rangeValues[axis] = childAxesStats[axis].values.filter(val => !isNaN(val));
		      }
		    });
		    detail.forEach((axis) => {
		      if (!colorValues[axis]) {
		        colorValues[axis] = [];
		      }
		      colorValues[axis].push(point.coordinates[axis]);
		    });
		  });

		 const axesData = {
		 parentAxesStats,
		 childAxesStats,
		 fullAxes,
		 mainDomain,
		 rangeAxes,
		 domainValues,
		 rangeValues,
		 colorValues,
		 indices
		 };

		 return {
		 points,
		 progressOptions,
		 interpType,
		 method,
		 range,
		 domain,
		 detail,
		 axesData,
		 };
		 }
	async processInheritedData({
		  points,
		  progressOptions,
		  interpType,
		  method,
		  range,
		  domain,
		  detail,
		  axesData
		}) {
		  const [ parentAxesStats, childAxesStats, parentIndices, childIndices ] = [ axesData.parentAxesStats, axesData.childAxesStats, axesData.indices.parent, axesData.indices.child ];
		  
		
		
		 // Simple inheritance logic - fill in null values in child with parent values
		  if (parentIndices.length == childIndices.length && method && method.mappingType === 'simple' && method.dataInheritance === 'simple' && method.detailInheritance === 'simple') {
			  //////////////////////console.warn("SYNCHRONOUS INHERITENCE");
		    const resultPoints = points.map((point, i) => {
		      const newCoordinates = {};
		      //////////////////////////////console.log(point);
		      Object.entries(point.coordinates).forEach(([axis, value]) => {
		        if (value === null) {
		          if (this.points[i] && this.points[i].coordinates[axis] !== null) {
		            newCoordinates[axis] = this.points[i].coordinates[axis];
		          } else {
		            newCoordinates[axis] = null;
		          }
		        } else {
		          newCoordinates[axis] = value;
		        }
		      });
		      return new Point(newCoordinates, this.points[i]);
		    });
		    return {
		      points: resultPoints,
		      progressOptions,
		      interpType: interpType,
		      method: method,
		      range: range,
		      domain: domain,
		      detail: detail,
		      axesData: axesData,
		    };
		  }
		
		  let mainInterpolator;
		  const mainDomain = axesData.mainDomain;
		  const domainValues = axesData.domainValues;
		  const rangeAxes = axesData.rangeAxes;
		  const rangeValues = axesData.rangeValues;
		  const colorValues = axesData.colorValues;
		  const indices = axesData.indices;

		  if (
  parentIndices.length !== childIndices.length &&
  method &&
  method.mappingType === "simple" &&
  method.dataInheritance === "simple" &&
  method.detailInheritance === "simple"
) {
  /*
  const parentPointCollection = new PointCollection(this.points);
  const childPointCollection = new PointCollection(points);

  await Promise.all([
    parentPointCollection.calculationPromise,
    childPointCollection.calculationPromise,
  ]);

  const parentAxesStats = parentPointCollection.axisStats;
  const childAxesStats = childPointCollection.axisStats;
*/
//////console.warn(parentAxesStats, childAxesStats);

  const fullAxes = Array.from(
    new Set([
      ...Object.keys(parentAxesStats).filter(
        (axis) => parentAxesStats[axis].values.every((val) => val != null)
      ),
      ...Object.keys(childAxesStats).filter(
        (axis) => childAxesStats[axis].values.every((val) => val != null)
      ),
    ])
  );
////////console.warn(fullAxes);
  const ratio = parentIndices.length / childIndices.length;

  const parentAxesData = {};
  const childAxesData = {};
  fullAxes.forEach((axis) => {
    let parentValues;
    if (parentAxesStats[axis]) {
      parentValues = parentAxesStats[axis].values.map((value, index) => ({
        index: index / ratio,
        value,
      }));
    } else {
      parentValues = childAxesStats[axis].values.map((value, index) => ({
        index,
        value,
      }));
    }
    let childValues;
    if (childAxesStats[axis]) {
      childValues = childAxesStats[axis].values.map((value, index) => ({
        index,
        value,
      }));
    } else {
      childValues = parentValues;
    }
    const domainValues = [
      ...new Set([
        ...parentValues.map(({ index }) => index),
        ...childValues.map(({ index }) => index),
      ]),
    ];
    const parentInterpolator = new Interpolator(
      parentValues,
      domainValues,
      interpType
    );
    const childInterpolator = new Interpolator(
      childValues,
      domainValues,
      interpType
    );
    parentAxesData[axis] = { domainValues, interpolator: parentInterpolator };
    childAxesData[axis] = { domainValues, interpolator: childInterpolator };
  });

  const resultPoints = points.map((point) => {
    const newCoordinates = {};
    const scaledChildIndex = Math.floor(point.coordinates["i"] * ratio);
    
    Object.entries(point.coordinates).forEach(([axis, value]) => {
      if (value === null && parentIndices[scaledChildIndex]) {
        const parentCoord = this.points[parentIndices[scaledChildIndex]].coordinates;
        if (parentCoord[axis] !== null) {
          newCoordinates[axis] = parentCoord[axis];
        } else {
          const domainValue = point.index * ratio;
          if (parentAxesData[axis] && parentAxesData[axis].interpolator) {
            const interpolatedValue = parentAxesData[axis].interpolator.evaluate(
              domainValue
            );
            newCoordinates[axis] = interpolatedValue;
          } else {
            newCoordinates[axis] = value;
          }
        }
      } else {
        newCoordinates[axis] = value;
      }
    });
    //////console.warn(points);
//////console.warn(newCoordinates, this.points, parentIndices);
//////console.warn(this.points[parentIndices[scaledChildIndex]]);
    return new Point(newCoordinates, this.points[parentIndices[scaledChildIndex]]);
  });
return {
  points: resultPoints,
  interpType: interpType,
  method: method,
  range: range,
  domain: domain,
  detail: detail,
  axesData: {
    indices: { parentIndices, childIndices },
    mainDomain,
    domainValues,
    rangeAxes,
    rangeValues,
    colorValues,
  },
};
}
		  if (mainDomain === 'index') {
		    mainInterpolator = new Interpolator(indices, indices, interpType);
		  } else {
		    mainInterpolator = new Interpolator(domainValues, domainValues, interpType);
		  }

		  const rangeInterpolators = {};
		  rangeAxes.forEach((axis) => {
		    rangeInterpolators[axis] = new Interpolator(domainValues, rangeValues[axis], interpType);
		  });

		  const colorInterpolators = {};
		  detail.forEach((axis) => {
		    colorInterpolators[axis] = new Interpolator(domainValues, colorValues[axis], interpType);
		  });

		  const resultPoints = points.map((point) => {
		    const domainValue = point.coordinates[mainDomain];
		    if (domainValue === undefined || isNaN(domainValue)) {
		      return point.clone();
		    }

		    const interpolatedPoint = new Point({});
		    interpolatedPoint.coordinates[mainDomain] = domainValue;

		    rangeAxes.forEach((axis) => {
		      const interpolatedValue = rangeInterpolators[axis].evaluate(domainValue);
		      if (interpolatedValue !== undefined) {
		        interpolatedPoint.coordinates[axis] = interpolatedValue;
		      } else {
		        interpolatedPoint.coordinates[axis] = point.coordinates[axis];
		      }
		    });

		    detail.forEach((axis) => {
		      const interpolatedValue = colorInterpolators[axis].evaluate(domainValue);
		      if (interpolatedValue !== undefined) {
		        interpolatedPoint.coordinates[axis] = interpolatedValue;
		      } else {
		        interpolatedPoint.coordinates[axis] = point.coordinates[axis];
		      }
		    });

		    return interpolatedPoint;
		  });

		  return {
		    points: resultPoints,
		    interpType: interpType,
		    method: method,
		    range: range,
		    domain: domain,
		    detail: detail,
		    axesData: axesData,
		  };
		}
	 fillChildAxes({
		  points,
		  interpType,
		  method,
		  range,
		  domain,
		  detail,
		  axesData
		}){
		 //////////////////////////////////console.log(points);		 
		 return points;
	 }
	 
		  returnAxes(axes, selectFlag = true) {
			  if (selectFlag) {
			    // Return each axis listed in axes as an array of values of each point in index order.
			    const result = {};
			    for (const axis of axes) {
			      result[axis] = this.points.map((point) => point.coordinates[axis]);
			    }
			    return result;
			  } else {
			    // Return each axis present in the dataset but absent in axes.
			    const allAxes = Object.keys(this.points[0].coordinates);
			    const result = {};
			    for (const axis of allAxes) {
			      if (!axes.includes(axis)) {
			        result[axis] = this.points.map((point) => point.coordinates[axis]);
			      }
			    }
			    return result;
			  }
			}
setInterpolator(type = 'linear', customInterpolator = null) {
  const i = this.points.map((p) => p.coordinates.i);
  const ranges = Object.keys(this.points[0].coordinates)
    .filter((key) => key !== 'i')
    .map((key) => this.points.map((p) => p.coordinates[key]));

  this.interpolator = new Interpolator(i, ranges);
  this.interpolator.setType(type);
  if (customInterpolator) {
    this.interpolator.setCustomInterpolator(customInterpolator);
  }
}
			  
			  getInterpolator(){
				  return this.interpolator;
			  }

			  addPoint(input) {
				  this.id = null;
				  //////////////////////////////////////console.trace();
				  ////////////////////////////////////////console.log(input);
				    let point;
				    if (input instanceof Point) {
				      point = input;
				    } else if (typeof input === 'object' && input.x && input.y) {
				      point = new Point(input);
				    } else {
				      ////////////////////////////////////console.error('Invalid input. Expected a Point or an object with x and y properties.');
				      return;
				    }
					let interpolator = null;
				    this.points.push(point);
				    if(this.points.length >= 3){
				    	if(this.interpolator){
				    this.interpolator.resetInterpolator(this.points.map((p) => p.coordinates.x), this.points.map((p) => p.coordinates.y));
				    }else{
				    	this.interpolator = new Interpolator(this.points.map((p) => p.coordinates.x), this.points.map((p) => p.coordinates.y));
				    }
				    interpolator = this.interpolator;
				    }
				    if(this.points.length > 1){
				    this.intervals = DataProcessor.getIntervalsFromPoints(this.points, interpolator);
				    return this.intervals[this.intervals.length - 1];
				    }
				    return null;
				}
replacePointsFast(points){
	if (points[0] && (!points[0].coordinates['i'] || points[0].coordinates.i == null)){
	  for (let i = 0; i < points.length; i++) {
    points[i].coordinates.i = i;
  }
  }
	this.points = points;
}
static async indexPoints(points, chainlink, myID) {
	console.log("indexing");

	if(!myID){
		myID = chainlink?.inputControl?.progressBarManager?.getID() || Math.random();
	}
  const worker = new MyWorker(
    chainlink,
    async (i) => {
	//	console.log(i);
      const point = points[i];

      if (!point.coordinates["i"] || point.coordinates.i == null) {
        point.coordinates.i = i;
      }
//console.warn(chainlink.mapping);
      if (mappingsArray) {
        const discardedAxes = {};
//console.warn(mappingsArray);
        for (const [input, output] of mappingsArray) {
          // Store the overwritten output value in discardedAxes
          discardedAxes[output] = point.coordinates[output];

          // Map the input value to the output
          point.coordinates[output] = (point.coordinates[input] !== undefined && point.coordinates[input] !== null) ? point.coordinates[input] : discardedAxes[input];
        }
      }

      return point;
    },
    "Indexing Data",
    points.length,
    0,
    myID
  );
let mappingsArray;
  if (chainlink?.mapping) {
    mappingsArray = chainlink.mappings['output'].split(',').map((pair) => pair.split(':'));
  }

  const indexedPoints = await worker.runThreaded({points, mappingsArray}, ['Point']);
  return indexedPoints;
}

async replacePoints(points, chainlink = null, mapping = null) {
const myID = chainlink?.inputControl?.progressBarManager?.getID() || Math.random();
  const progress = chainlink
    ? (id, value, label) => chainlink.inputControl.progress(id, value, label)
    : () => {};
if(!this.points){
	this.points = points;
	return;
}


  if (!points) {
    this.points = [];
    return;
  }
//////console.warn(points);
 points = await DataSet.indexPoints(points, chainlink, myID, mapping);
//////console.warn(points);

  const prunedPoints = [];
  let seenPoints;
  const sigfig = this.sigfig;

  const duplicateWorker = new MyWorker(
    chainlink,
    async (k) => {
	//	//////console.warn("Checking Duplicates in ", points);
  const point = points[k];
  ////////console.log(`Current point: ${JSON.stringify(point)}`);

  if(!seenPoints){
    seenPoints = new Set();
    ////////console.log(`Initialized seenPoints Set: ${seenPoints}`);
  }
  
  // Check for exact duplicate
  if (seenPoints.has(JSON.stringify(point))) {
    //console.log(`Found exact duplicate: ${JSON.stringify(point)}`);
    return null;
  }

  // Check for key-value duplicate with different order
  const sortedCoordinates = {};
  if (point.coordinates) {
    for (const key of Object.keys(point.coordinates).sort()) {
      if (point.coordinates[key]) {
        sortedCoordinates[key] = parseFloat(
          point.coordinates[key].toFixed(sigfig)
        );
      } else {
        sortedCoordinates[key] = point.coordinates[key];
      }
    }
    const sortedPoint = new Point(sortedCoordinates);
   // //////console.log(`Sorted point: ${JSON.stringify(sortedPoint)}`);
    if (seenPoints.has(JSON.stringify(sortedPoint))) {
      //console.log(`Found key-value duplicate with different order: ${JSON.stringify(sortedPoint)}`);
      return null;
    }
    seenPoints.add(JSON.stringify(point));
    seenPoints.add(JSON.stringify(sortedPoint));
   // //////console.log(`Added point to seenPoints Set: ${seenPoints}`);
    return point;
  } else {
    //console.log(`Point has no coordinates`);
    return null;
  }
},
    "Checking For Duplicates",
    points.length,
    0, myID
  );
//////console.warn(this.points);
  const postDupCheck = await duplicateWorker.runThreaded({'sigfig':sigfig, 'points': points, 'seenPoints':seenPoints, 'standardAxes':standardAxes}, ['Point']);
  //////console.log(postDupCheck);
  
  const prunedPointsTemp = postDupCheck.filter((point) => point !== null);
  this.points = await this.inheritAxesData(prunedPointsTemp, {chainlink, id:myID});
////console.warn(this.points);
 this.points = await DataSet.indexPoints(this.points, chainlink, myID, mapping);


  let domainValues = this.points.map((p) => p.coordinates.i);
  let rangeKeys = Object.keys(this.points[0].coordinates).filter(
    (key) => key !== "i" && key !== "none"
  );

  let rangeValues = [];
  for (let j = 0; j < rangeKeys.length; j++) {
    rangeValues.push(this.points.map((p) => p.coordinates[rangeKeys[j]]));
  }

  if (!this.interpolator) {
    this.interpolator = new Interpolator(domainValues, rangeValues);
  } else {
    this.interpolator.resetInterpolators(domainValues, rangeValues);
    this.interpolator.setType(this.interpolator.type);
  }
  this.intervals = DataProcessor.getIntervalsFromPoints(
    prunedPoints,
    this.interpolator
  );
}
			 
				createIntervals(points, powerThreshold, timeThreshold, minInterval) {
					//////////////////////////////////////console.trace();
					////////////////////////////////////////console.log(points);
				    const intervals = MathematicalOperation.createIntervals(points, powerThreshold, timeThreshold, minInterval);
				    return intervals;
				}
			  addInterval(input) {
				  this.id= null;
			    let interval;
			    if (input instanceof Interval) {
			      interval = input;
			    } else if (typeof input === 'object' && input.start && input.end && input.data) {
					//////////console.trace();
			      interval = new Interval(input);
			    } else {
			      ////////////////////////////////////console.warn('Invalid input. Expected an Interval or an object with start, end, and data properties.');
			     // return;
			    }
			    const convertedPoints = MathematicalOperator.convertIntervalsToPoints([interval]);
			    this.intervals.push(interval);
			    this.points = this.points.concat(convertedPoints);
			  }

	  getPointAtTime(time) {
	    // implementation
	  }

	  getIntervalAtTime(time) {
	    // implementation
	  }

	  getAllX() {
////////////////////////////////////////console.log(JSON.stringify(this.points));
		    return this.points.map(point => point.coordinates['x']);
		  }

		  getAllY() {
		    return this.points.map(point => point.coordinates['y']);
		  }

		  getAllD() {
			//  //////////////////////////////////////console.log(this.intervals.map(interval => interval.getDuration()));
		    return this.intervals.map(interval => interval.getDuration());
		  }
		  getAllP() {
			    return this.intervals.map(interval => interval.getPower());
			  }
		  getAllE() {
			    return this.intervals.map(interval => interval.getEnd());
			  }
		  getAllS() {
			    return this.intervals.map(interval => interval.getStart());
			  }
static clone(dataset) {
  let newPoints = [];
  let newIntervals = [];

  if (dataset.points && dataset.points.length) {
    newPoints = dataset.points.map((point) => {
      return new Point(point.coordinates);
    });
  }

  if (dataset.intervals && dataset.intervals.length) {
	  //////////console.error(dataset.intervals);
    newIntervals = dataset.intervals.map((interval) => {
      const point = new Point([interval.getStart(), interval.getPower()]);
      return new Interval(interval.start, interval.end, point);
    });
  }

  if (newPoints.length || newIntervals.length) {
    return new DataSet(newPoints, newIntervals);
  } else {
    return new DataSet();
  }
}
	  reduce() {
	    const initial = { duration: 0, power: 0 };
	    const reducer = (accumulator, interval) => {
	      return {
	        duration: accumulator.duration + interval.getDuration(),
	        power: accumulator.power + interval.getPower()
	      };
	    };
	    return this.intervals.reduce(reducer, initial);
	  }
	  getNormalizedAxis(axisLabel) {
  const axisValues = this.points.map(point => point.coordinates[axisLabel]);
  const maxValue = Math.max(...axisValues);
  const normalizedValues = axisValues.map(value => value / maxValue);
  return normalizedValues;
}
prune(everyNthPoint) {
	const maxPrunedSize = 1000/everyNthPoint;
	const numberToDrop = this.points.length / parseFloat(maxPrunedSize);
	if(numberToDrop>1 &&this.points.length > 3 * numberToDrop){
	//////////////////////////console.error(this.points);
  let points = this.points;
  let prunedPoints = [];
  for (let i = 0; i < points.length; i += numberToDrop) {
	//////////////////////////console.error(i, points[i]);
	      prunedPoints.push(points[Math.floor(i)]);
  }
  this.replacePoints(prunedPoints);
  }
  return this;
}
	  static compare(firstDataset, secondDataset) {
		  
		  
		 // return (JSON.stringify(firstDataset) == JSON.stringify(secondDataset));
		  
		    if (firstDataset.id && secondDataset.id && firstDataset.id === secondDataset.id) {
		      return true;
		    }
		    const uuid = () => Math.random().toString(36).substring(2) + Date.now().toString(36);

		    const firstPoints = firstDataset.points;
		    const secondPoints = secondDataset.points;
		    
		    if(!firstPoints || !secondPoints || !firstPoints[0]?.coordinates || !secondPoints[0]?.coordinates){
				////console.error(firstDataset, secondDataset)
		    	return false; 
		    }

		    const axesOne = Object.keys(firstPoints[0].coordinates);
		    const axesTwo = Object.keys(secondPoints[0].coordinates);
		    if (axesOne.length !== axesTwo.length || !axesOne.every(axis => axesTwo.includes(axis))) {
		      return false;
		    }
		    const axes = axesOne;

		    for (let i = 0; i < firstPoints.length; i++) {
		      const firstPoint = firstPoints[i];
		      const secondPoint = secondPoints[i];
		      for (const axis of axes) {
		        if ((!firstPoint && secondPoint) || (firstPoint && !secondPoint) || firstPoint.coordinates[axis] !== secondPoint.coordinates[axis]) {
		          return false;
		        }
		      }
		    }

		    const firstIntervals = firstDataset.intervals;
		    const secondIntervals = secondDataset.intervals;
		    if(!(!firstIntervals && !secondIntervals)){
		    if ((firstIntervals && !secondIntervals) || (!firstIntervals && secondIntervals) || ((firstIntervals && secondIntervals)&&(firstIntervals.length !== secondIntervals.length))) {
		      return false;
		    }
		    for (let i = 0; i < firstIntervals.length || 0; i++) {
		      const firstInterval = firstIntervals[i];
		      const secondInterval = secondIntervals[i];
		      if (firstInterval.duration !== secondInterval.duration || firstInterval.power !== secondInterval.power) {
		        return false;
		      }
		    }
		    }
		    
		    //////////////////////////console.warn(firstDataset, secondDataset);

		    if (firstDataset.id === undefined) {
		      firstDataset.id = uuid();
		    }
		    if (secondDataset.id === undefined) {
		      secondDataset.id = uuid();
		    }
		    if (firstDataset.id !== secondDataset.id) {
		      secondDataset.id = firstDataset.id;
		    }

		    return true;
		  }
		
	}


class DataProcessor {
	constructor(dataSet, interpolationType = 'linear') {
	this.activeDataSets = [];
	this.addDataSet(dataSet);
	this.cacheDataSets = [];
	this.cacheDataSetsLength = 100;
	this.interpolationTypa = interpolationType;
	}
	static createDataSetFromPointsMap(data, interpolator) {
		  const points = convertDataStructure(data, 'points');
		  const dataSet = new DataSet(points);
		  if (interpolator) {
		    interpolator.setData(dataSet);
		    dataSet.interpolator = interpolator;
		  }
		  return dataSet;
		}

		static createDenseMapFromDataSet(dataSet) {
		  const interpolator = dataSet.interpolator || new Interpolator();
		  interpolator.setData(dataSet);
		  return interpolator.getDenseMap();
		}
		static sortMapToArrays(map) {
			  const [xValues, yValues] = Object.entries(map)
			    .map(([key, value]) => [parseFloat(key), value])
			    .sort(([a], [b]) => a - b)
			    .reduce(([xs, ys], [x, y]) => [[...xs, x], [...ys, y]], [[], []]);
			  return [xValues, yValues];
			}
		static convertDataStructure(data, targetStructure) {
		  if (targetStructure === 'points') {
		    if (Array.isArray(data)) {
		      // Convert from an array of objects to an array of Point instances
		      return data.map(p => new Point([p.x, p.y]));
		    } else if (data instanceof WorkoutPlan) {
		      // Convert from a WorkoutPlan instance to an array of Point instances
		      return data.intervals.map(i => new Point([i.getDuration(), i.getPower()]));
		    }
		  } else if (targetStructure === 'intervals') {
		    if (Array.isArray(data)) {
		      // Convert from an array of Point instances to an array of Interval instances
		      const intervals = [];
		      for (let i = 0; i < data.length - 1; i++) {
		        const interval = new Interval(null, null, data[i]);
		        interval.setEnd(data[i+1].coordinates.x);
		        intervals.push(interval);
		      }
		      return intervals;
		    } else if (data instanceof DataSet) {
		      // Convert from a DataSet instance to an array of Interval instances
		      return data.intervals;
		    }
		  } else if (targetStructure === 'workoutPlan') {
		    if (Array.isArray(data)) {
		      // Convert from an array of Point instances to a WorkoutPlan instance
		      const intervals = data.map(p => new Interval(p.coordinates.x, null, p));
		      return new WorkoutPlan(intervals);
		    } else if (data instanceof DataSet) {
		      // Convert from a DataSet instance to a WorkoutPlan instance
		      return new WorkoutPlan(data.intervals);
		    }
		  } else if (targetStructure === 'dataSet') {
		    if (Array.isArray(data)) {
		      // Convert from an array of Point instances to a DataSet instance
		      const intervals = data.map((p, i, arr) => {
		        if (i === 0 || i === arr.length - 1) {
		          return null;
		        }
		        const interval = new Interval(null, null, p);
		        interval.setEnd(arr[i+1].coordinates.x);
		        return interval;
		      }).filter(i => i !== null);
		      const dataSet = new DataSet(data, intervals);
		      if (dataSet.interpolator) {
		        dataSet.interpolator.setData(dataSet);
		      }
		      return dataSet;
		    } else if (data instanceof WorkoutPlan) {
		      // Convert from a WorkoutPlan instance to a DataSet instance
		      const points = data.intervals.map(i => new Point([i.getDuration(), i.getPower()]));
		      const intervals = points.map((p, i, arr) => {
		        if (i === 0 || i === arr.length - 1) {
		          return null;
		        }
		        const interval = new Interval(null, null, p);
		        interval.setEnd(arr[i+1].coordinates.x);
		        return interval;
		      }).filter(i => i !== null);
		      const dataSet = new DataSet(points, intervals);
		      if (dataSet.interpolator) {
		        dataSet.interpolator.setData(dataSet);
		      }
		      return dataSet;
		    }
		  }
		}

		
		static getPointsFromIntervals(intervals, numPoints = 3) {
		    const points = [];
		    for (const interval of intervals) {
		        const { start, end, power } = interval;
		        const intervalSize = end - start;
		        const step = intervalSize / (numPoints - 1);
		        for (let i = 0; i < numPoints; i++) {
		            const xVal = start + i * step;
		            const yVal = power;
		            points.push({ coordinates: { x: xVal, y: yVal } });
		        }
		    }
		    return points;
		}
		
		/*
	  static getPointsFromIntervals(intervals, step = .01, interpolator = null) {
		    let x = [];
		    let y = [];
		    for (let i = 0; i < intervals.length; i++) {
		      let interval = intervals[i];
		      let start = interval.start;
		      let end = interval.end;
		      for (let j = start; j <= end; j += step) {
		        x.push(j);
		        y.push(interval.power);
		      }
		    }
		    if (!interpolator) {
		      interpolator = new Interpolator(x, y, 'linear');
		    }
		    let uniqueX = [];
		    let uniqueY = [];
		    let visited = {};
		    for (let i = 0; i < x.length; i++) {
		      if (!visited[x[i]]) {
		        visited[x[i]] = true;
		        uniqueX.push(x[i]);
		        uniqueY.push(y[i]);
		      }
		    }
		    let points = [];
		    for (let i = 0; i < uniqueX.length; i++) {
		      let xVal = uniqueX[i];
		      let yVal = interpolator.evaluate(xVal);
		     // //////////////////////////////////////console.log("interpolator returned: " + yVal);
		      points.push({ coordinates: { x: xVal, y: yVal } });
		    }
		    ////////////////////////////////////////console.log("points from interpolation of interval: " + JSON.stringify(points));
		    return points;
		  }*/

	  static breakArrays(interpolator, start, span, threshold=.1) {
		  const step = 0.1; // step size for evaluating the interpolator

		  // Get the original x and y arrays of the interpolator
		  const xArr = interpolator.domain;
		  const yArr = interpolator.ranges[0];
		  //return [[xArr, yArr]];
////////////////////////////////////////console.log(xArr);
		  // Initialize arrays for interpolated x and y values
		  const intXArr = [];
		  const intYArr = [];

		  // Evaluate the interpolator at regular intervals
		  for (let x = start; x < start + span; x += step) {
		    const y = interpolator.evaluate(x);
		    intXArr.push(x);
		    intYArr.push(y);
		  }
		  ////////////////////////////////////////console.log(intXArr);

		  // Find discontinuities based on threshold
		  const breakIndices = [];
		  let prevY = intYArr[0];
		  for (let i = 1; i < intYArr.length; i++) {
		    const y = intYArr[i];
		    if (Math.abs(y - prevY) > threshold) {
		      // discontinuity
		      breakIndices.push(i);
		    }
		    prevY = y;
		  }

		  // Find corresponding x values in the original array for each breakpoint
		  const breakpoints = breakIndices.map((index) => {
		    const intX = intXArr[index];
		    let xIndex = 0;
		    while (xIndex < xArr.length && xArr[xIndex] < intX) {
		      xIndex++;
		    }
		    return xArr[xIndex];
		  });

		  // Split arrays at breakpoints
		  const arrays = [];
		  let prevIndex = 0;
		  breakpoints.forEach((breakpoint) => {
		    const index = xArr.indexOf(breakpoint);
		    arrays.push([ xArr.slice(prevIndex, index), yArr.slice(prevIndex, index) ]);
		    prevIndex = index;
		  });
		  arrays.push([ xArr.slice(prevIndex), yArr.slice(prevIndex)]);

	//	  //////////////////////////////////////console.log(arrays);
		  // Return the arrays of x and y values for each span
		  return arrays;
		}
		
	  static getIntervalsFromPoints(points, interpolator, powerCalculation = 'integral', step = 0.01, minInterval = 0.1, powerThreshold = .1) {  
		  let newIntervals = [];
		    let numPoints = points.length;
		    let x = points.map(point => point.coordinates.x);
		    let y = points.map(point => point.coordinates.y);
		    let start = Math.min(...x);
		    
		    let end = Math.max(...x);
		    step = (end - start)*step; 
		    let snapToValue = MathematicalOperation.snapToValue;

		    if (numPoints == 1){
		    	////////////////////////////////////console.error("trying to make a one point interval");
		    	return;
		    }
		    
		    // create a third point if fewer than 3 points were provided
		    if (numPoints == 2) {
		        let avgX = (x[0] + x[1]) / 2;
		        let avgY = (y[0] + y[1]) / 2;
		        let oldX = x[x.length - 1];
		        let oldY = y[y.length - 1];
		        x[x.length - 1] = avgX;
		        y[y.length - 1] = avgY;
		        x.push(oldX);
		        y.push(oldY);
		        //sort x and y here
		        numPoints = 3;
		    }

		    // initialize an interpolator if one is not provided
		    if (!interpolator) {
		        interpolator = new Interpolator(x, y);
		//        //////////////////////////////////////console.log("New Interpolator Initialized");
		    }else{
	//	    	//////////////////////////////////////console.log("Interpolator Already Present");
		    }

		    //let denseMap = interpolator.createDenseMap();
		    ////////////////////////////////////////console.log(denseMap);
		    //if(!denseMap){
		    //	////////////////////////////////////console.error("denseMap came back empty");
		    //}
		    let newX = [];
		    let newY = [];

		    //let values = this.extractDenseMapValues(denseMap)
		    newX = x;//values[0];
		    newY = y;//values[1];
		    //interpolator.resetInterpolators(newX, newY);
		    let fullStart = start;
		    let fullEnd = end;
		    let segmented = this.breakArrays(interpolator, start, end - start);
		   // //////////////////////////////////////console.log(segmented);
		    let leftoverX = [];
		    let leftoverY = [];
		    interpolator.resetInterpolators(newX, [newY]);
		    for(let k =0; k < segmented.length; k++){
		    	if(segmented[k][0].length <= 2){
		    		for(let j = 0; j<segmented[k][0].length; j++){
		    			leftoverX.push(segmented[k][0][j]);
		    			leftoverY.push(segmented[k][1][j]);
		    		}
		    	 continue;	
		    	}
		    	newX = leftoverX;
		    	newY = leftoverY;
		    	
		    	
				for(let j = 0; j < segmented[k][0].length; j++){
			    	newX.push(segmented[k][0][j]);
			    	newY.push(segmented[k][1][j]);
		
				}
				leftoverX = [];
				leftoverY = [];

		    
//		    ////////////////////////////////////console.error(interpolator);
//		    ////////////////////////////////////console.trace();
		    
		   start = snapToValue(segmented[k][0][0], [1]);
		   end = snapToValue(segmented[k][0][segmented[k][0].length - 1], [1]);
		    
		    let powerFunction = interpolator.evaluate.bind(interpolator);
		    let integralFunction = interpolator.integral.bind(interpolator);

		    let currentInterval;
		    let numOutputPoints = (end - start )/step;
		    let currentIntervalLength = 0;
		    let numPointsInInterval = 0;
		    let currentIntervalStart = start;
		    let currentIntervalEnd = start;
		    let currentIntervalPower = 0;
		    for (let i = 0; i < numOutputPoints; i++) {
		        
				let powerAtNewPoint = powerFunction(start+i*step);
				//////////////////////////////////////console.warn(powerAtNewPoint);
				if (currentIntervalLength >= minInterval && Math.abs(powerAtNewPoint - currentIntervalPower) >= powerThreshold) {
					  let snappedEnd = Math.round(snapToValue(currentIntervalEnd, [10, 5, 1])*100)/100;
					  if(snappedEnd > end){
						  snappedEnd = end;
					  }
					  let snappedDuration = snappedEnd - currentIntervalStart;
					  let snappedPower = snapToValue(currentIntervalPower, [10, 5, 1]);
					  
					  currentInterval = new Interval(currentIntervalStart, snappedEnd, new Point([snappedDuration, snappedPower]));
					  newIntervals.push(currentInterval);
					  currentIntervalStart = snappedEnd;
					  currentIntervalPower = 0;
					  numPointsInInterval = 0;
					  currentIntervalLength = 0;
					} else {
					  currentIntervalLength += step;
					  currentIntervalEnd += step;
					  if(currentIntervalEnd > end){
						  let overrun = currentIntervalEnd - end;
						  currentIntervalEnd -= overrun;
						  currentIntervalLength -= overrun;
					  }
					  if (currentIntervalLength > 0) {
						////////////////////////////////////////console.log("integrating with values: "+currentIntervalStart+":"+currentIntervalEnd+":"+currentIntervalLength);
					    let integral = integralFunction(currentIntervalStart, currentIntervalEnd) / currentIntervalLength;
						if(!isNaN(integral)){ 
							currentIntervalPower = integral; 
						}else{
							//////////////////////////////////////console.error("NaN INTEGRAL");
						}
					  	////////////////////////////////////////console.log("currentIntervalPower: "+currentIntervalPower);
					  } else {
					    currentIntervalPower = evaluateFunction(currentIntervalStart);
					    if(!currentIntervalPower){
					    	currentIntervalPower = 0;
					    }
					  }
					  numPointsInInterval++;
					}
		        
		    }

		    // handle last interval
		    if (currentIntervalStart < end && currentIntervalLength > 0) {
  let snappedEnd = snapToValue(currentIntervalEnd, [10, 5, 1]);
  let snappedDuration = Math.round((snappedEnd - currentIntervalStart)*100)/100
  let integral = integralFunction(currentIntervalStart, currentIntervalEnd) / currentIntervalLength;

  //////////////////////////////////////console.warn("Integral, start, end, length: "+ integral +":"+currentIntervalStart+":"+ currentIntervalEnd+":"+ currentIntervalLength);
	if(!isNaN(integral)){ 
		currentIntervalPower = integral; 
	}
	//////////////////////////////////////console.warn("currentIntervalPower: "+currentIntervalPower);
  let snappedPower = Math.round(snapToValue(currentIntervalPower,[10, 5, 1])*100)/100;
  ////////////////////////////////////////console.log(snappedPower +":"+currentIntervalPower);
  
  if (currentInterval && currentIntervalLength < minInterval) {
    currentIntervalStart = currentInterval.getStart();
    currentInterval = new Interval(currentIntervalStart, snappedEnd, new Point([snappedDuration, snappedPower]));
    newIntervals[newIntervals.length - 1] = currentInterval;
  } else if (!currentInterval) {
    currentIntervalStart = start;
    currentInterval = new Interval(currentIntervalStart, snappedEnd, new Point([snappedDuration, snappedPower]));
    newIntervals.push(currentInterval);
  } else {
    currentInterval = new Interval(currentIntervalStart, snappedEnd, new Point([snappedDuration, snappedPower]));
    newIntervals.push(currentInterval);
  }
}
	  }
////////////////////////////////////////console.log(newIntervals);
interpolator.resetInterpolators( x, [y]);
		    return newIntervals;
		}
	  static getDataSetFromIntervals(intervals){
		  ////////////////////console.log(JSON.stringify(this.getPointsFromIntervals(intervals)) +":"+ JSON.stringify(intervals));
	  
		  return new DataSet(this.getPointsFromIntervals(intervals), intervals);
	  }
	  static calculatePower(points, powerCalculation) {
		  const x = points.map(point => point.coordinates.x);
		  const y = points.map(point => point.coordinates.y);
		  const interpolator = new Interpolator(x, y);

		  switch (powerCalculation) {
		    case 'average':
		      return (y[0] + y[y.length - 1]) / 2;

		    case 'integral':
		      return interpolator.integral(x[0], x[x.length - 1], 20);

		    case 'rms':
		      return MathematicalOperation.rmsValues(points, 3).points[0].value;

		    default:
		      throw new Error(`Invalid power calculation method: ${powerCalculation}`);
		  }
		}
		  
	  static getXY(points, intervalStart = 0, intervalEnd = null) {
		  const filteredPoints = points.filter(point => point.coordinates.x >= intervalStart && (intervalEnd === null || point.coordinates.x <= intervalEnd));
		  const x = filteredPoints.map(point => point.coordinates.x - intervalStart);
		  const y = filteredPoints.map(point => point.coordinates.y);
		  return { x, y };
		}

	  getDataSetById(id){
		  return this.activeDataSets[findIndexById(id)];
		  
	  }
	  
	findIndexById(id){
		for(let i = 0; i < this.activeDataSets.length; i++){
			if(this.activeDataSets[i].id == id){
				return i;
			}
			
		}
	}
getEvalFunc(id = null, index = null){
	let dataSet;
	if(!id){
		if(index){
			dataSet = this.activeDataSets[index]; 
			
		}else{
			index = this.findIndexById(id);
		}
	}
	 
	return dataSet.interpolator.evaluate;
	
}
	processData(id = null, index = null) {
		if(!id){
		if(index){
			id = this.findIdByIndex(index);
		}else{
			index = this.findIndexById(id);
		}
	}
		const dataSet = this.getDataSet(index);
	dataSet.averagePower = this.calculateAveragePower(dataSet);
	this.calculateTotalEnergy(dataSet);
	this.calculateIntegralChart(dataSet);
	this.calculateDerivativeChart(dataSet);
	this.calculateAxisData(dataSet);
	}

	calculateAveragePower(dataSet) {
	return MathematicalOperation.averagePower(dataSet);
	}

	calculateTotalEnergy(dataSet) {
	return MathematicalOperation.totalWork(dataSet);
	}

	calculateIntegralChart(dataSet) {
	return MathematicalOperation.totalWork(dataSet);
	}

	calculateDerivativeMap(dataSet) {
	return MathematicalOperation.derivativeMap(dataSet);
	}

	calculateAxisData() {
	
	}

	addDataSet(dataSet) {
	this.activeDataSets.push(dataSet);
	}

	addCachedDataSet(dataSet) {
	if (this.cacheDataSets.length === this.cacheDataSetsLength) {
	this.cacheDataSets.shift();
	}
	this.cacheDataSets.push(dataSet);
	}

	getDataSets() {
	return this.activeDataSets;
	}

	getCachedDataSets() {
	return this.cacheDataSets;
	}

	removeCachedDataSet(dataSet) {
	const index = this.cacheDataSets.indexOf(dataSet);
	if (index > -1) {
	this.cacheDataSets.splice(index, 1);
	}
	}

	}