	// Class: Interpolator
	// This class provides a general interface for different interpolation algorithms
class Interpolator {
  constructor(domain, ranges) {
    this.domain = domain;
    for(let i = 0; i < domain.length; i++){
		if(domain[i] == null){
			//////////////////console.error("INVALID DOMAIN");
			//////////////////console.trace();
		}
	}
    //////////////////console.warn(domain);
    this.ranges = !ranges[0].length ? [ranges] : ranges;
    if(!ranges){
		////////////////////console.error("NO RANGE SUPPLIED");
	}
    
    for(let i = 0; i < this.ranges.length; i++){
		if(this.ranges[i].length < 3){
			////////////////////console.error("IVALID RANGE LENGTH");
		}
		for(let j = 0; j < this.ranges[i].length; j++){
			if(this.ranges[i][j] == null){
				               this.ranges.splice(i, 1);
                i--;
                break;
			}
		}
	}
    //////////////////console.error( this.ranges);
    //////////////////console.trace( ranges);
    this.type = 'linear';
    this.customInterpolator = null;
    this.interpolators = [];
    this.extrapolate = true;
    ////////////////////console.error(this.ranges);
    if(this.extrapolate){
		this._modifyDomain();
	}
    this.initializeInterpolators();
  }

_modifyDomain() {
	  // Sort the domain values in ascending order
	  ////////////////////console.warn(this.ranges);
let sortedIndices = this.domain.map((_, i) => i).sort((i, j) => this.domain[i] - this.domain[j]);
////////////////////console.warn(sortedIndices);
this.domain = sortedIndices.map(i => this.domain[i]);
this.ranges = this.ranges.map(r => sortedIndices.map(i => r[i]));
  ////////////////////console.warn(this.ranges);
  const eps = 1e-6;
  let newDomain = [];
  let newRanges = [];
  let i = 0;
  while (i < this.domain.length) {
    const x = this.domain[i];
    if(x == null){
		i++;
		continue;
	}
    const j = this.domain.lastIndexOf(x, i + 1);
     if (j === i) {
      // No duplicate domain values
      newDomain.push(x);
      ////////////////////console.warn(this.ranges.length, this.ranges);
      for(let k = 0; k < this.ranges.length; k++){
      const rangeValue = this.ranges[k][i];
      if(!rangeValue && rangeValue != 0){
		  ////////////////////console.error("INVALID RANGE VALUE AT INDEX ", k, i, this.ranges[k]);
		  return;
	  }
      if(newRanges.length-1 < k){
		  ////////////////////console.error(newRanges.length, k);
		  newRanges.push([]);
		  newRanges[k].push(rangeValue);
		  ////////////////////console.log(newRanges);
	  }else{
      newRanges[k].push(rangeValue);
      }}
      i++;
    } else {
      // Duplicate domain values found
      const xDoubleLeft = x - 2*eps;
      const xLeft = x - eps;
      const xRight = x + eps;
      const xDoubleRight = x + 2*eps;
      

      // Find the left and right range values for each dimension
      const leftValues = this.ranges.map((r, k) => {
		  
        const leftInterpolator = new LinearInterpolation([xDoubleLeft, xLeft, x], [[r[i - 1],(r[i-1]+(r.slice(i, j+1).reduce((acc, val) => acc + val, 0)) / (j - i + 1))/2, (r.slice(i, j+1).reduce((acc, val) => acc + val, 0)) / (j - i + 1)]]);
        return leftInterpolator.evaluate(xLeft, { method: 'linear' })[0];
      });
      const rightValues = this.ranges.map((r, k) => {
        const rightInterpolator = new LinearInterpolation([x, xRight, xDoubleRight], [[(r.slice(i, j+1).reduce((acc, val) => acc + val, 0)) / (j - i + 1), (r[i+1]+(r.slice(i, j+1).reduce((acc, val) => acc + val, 0)) / (j - i + 1))/2, r[i + 1]]]);
        return rightInterpolator.evaluate(xRight, { method: 'linear' })[0];
      });
/*
      // Construct the full range values for each dimension over the entire domain
      const domainRange = this.ranges.map((r, k) => {
        const interpolator = new LinearInterpolation([xDoubleLeft, xLeft, x, xRight, xDoubleRight], [[leftValues[k], (r.slice(i, j+1).reduce((acc, val) => acc + val, 0)) / (j - i + 1), rightValues[k]]]);
        return interpolator.evaluate(this.domain, { method: 'linear' });
      });
*/
      // Insert the left and right nudged points
      newDomain.push(xDoubleLeft, xLeft, x, xRight, xDoubleRight);
      for(let k = 0; k < newRanges.length; k++){
      	newRanges[k].push(this.ranges[k][i-1], leftValues[k], (this.ranges[k][j]+this.ranges[k][i])/2, rightValues[k], this.ranges[k][i+1]);
      }
      i = j + 1;
        sortedIndices = newDomain.map((_, i) => i).sort((i, j) => newDomain[i] - newDomain[j]);
  newDomain = sortedIndices.map(i => newDomain[i]);
  newRanges = newRanges.map(r => sortedIndices.map(i => r[i]));
    }
    
  }
this.domain = newDomain;
this.ranges = newRanges;

////////////////////console.error(this.domain, this.ranges);

}

initializeInterpolators() {
	////////////////////console.warn(this.domain, this.ranges);
    if (this.domain.length > 2) {
      if (this.customInterpolator) {
        return this.customInterpolator.evaluate(x);
      } else if (this.type === 'cubic') {
        this.cubicSpline = new CubicSplineInterpolation(this.domain, this.ranges);
        this.interpolators.push(this.cubicSpline);
      } else if (this.type === 'pchip') {
        this.pchip = new PchipInterpolation(this.domain, this.ranges);
        this.interpolators.push(this.pchip);
      } else if (this.type === 'bspline') {
        this.bspline = new BSplineInterpolation(this.domain, this.ranges);
        this.interpolators.push(this.bspline);
      } else if (this.type === 'wavelet') {
        this.wavelet = new WaveletInterpolation(this.domain, this.ranges, typeWavelets);
        this.interpolators.push(this.wavelet);
      } else if (this.type === 'linear') {
        this.linear = new LinearInterpolation(this.domain, this.ranges);
        this.interpolators.push(this.linear);
      } else if (this.type === 'akima') {
        this.akima = new AkimaInterpolation(this.domain, this.ranges);
        this.interpolators.push(this.akima);
      } else if (this.type.startsWith('poly')) {
        const degree = parseInt(this.type.slice(4), 10) || null;
        this.polynomial = new PolynomialInterpolator(degree, this.domain, this.ranges);
        this.interpolators.push(this.polynomial);
      } else if (this.type === 'catmull-rom') {
        this.catmullRom = new CatmullRomInterpolation(this.domain, this.ranges);
        this.interpolators.push(this.catmullRom);
      } else if (this.type === 'bezier') {
        this.bezier = new BezierInterpolator(this.domain, this.ranges);
        this.interpolators.push(this.bezier);
      }
    } else {
      ////////////////////////////console.error('INTERPOLATION FAILED:', this.domain);
    }
  }

	resetInterpolators(domain, ...ranges) {
		if (domain && ranges) {
			this.domain = domain;
			this.ranges = ranges.length === 1 ? ranges[0] : ranges;
		}
		this.initializeInterpolators();
	}

	evaluate(x, ...rest) {
		if (this.domain.length <= 2) {
			return;
		}
		if (this.customInterpolator) {
			return this.customInterpolator.evaluate(x);
		} else if (this.type === 'cubic') {
			if(this.ranges.length == 1){
				return this.cubicSpline.evaluate(x, ...rest)[0];
			}else{
				return this.cubicSpline.evaluate(x, ...rest);
			}
		} else if (this.type === 'pchip') {
			if(this.ranges.length == 1){
				return this.pchip.evaluate(x, ...rest)[0];
			}else{
				return this.pchip.evaluate(x, ...rest);
			}
		} else if (this.type === 'bspline') {
			if(this.ranges.length == 1){
				return this.bspline.evaluate(x, ...rest)[0];
			}else{
				return this.bspline.evaluate(x, ...rest);
			}
		} else if (this.type === 'wavelet') {
			if(this.ranges.length == 1){
				return this.wavelet.evaluate(x, ...rest)[0];
			}else{
				return this.wavelet.evaluate(x, ...rest);
			}
		} else if (this.type === 'linear') {
			if(this.ranges.length == 1){
				return this.linear.evaluate(x, ...rest)[0];
			}else{
				return this.linear.evaluate(x, ...rest);
			}
		} else if (this.type === 'akima') {
			if(this.ranges.length == 1){
				return this.akima.evaluate(x, ...rest)[0];
			}else{
				return this.akima.evaluate(x, ...rest);
			}
		} else if (this.type.startsWith('poly')) {
			if(this.ranges.length == 1){
				return this.polynomial.evaluate(x, ...rest)[0];
			}else{
				return this.polynomial.evaluate(x, ...rest);
			}
		} else if (this.type === 'catmull-rom') {
			if(this.ranges.length == 1){
				return this.catmullRom.evaluate(x, ...rest)[0];
			}else{
				return this.catmullRom.evaluate(x, ...rest);
			};
		} else if (this.type === 'bezier') {
			if(this.ranges.length == 1){
				return this.bezier.evaluate(x, ...rest)[0];
			}else{
				return this.bezier.evaluate(x, ...rest);
			}
		} else {
			throw new Error(`Invalid type specified. Type should be either "cubic", "pchip", "bspline", "wavelet", "linear", "akima", "poly", or "catmull-rom", got ${this.type}.`);
		}
	}


evaluateND(...args) {
	if (args.length !== this.ranges.length) {
		throw new Error('Number of arguments does not match number of ranges');
	}
	const yVals = this.ranges.map((range, i) => {
		return this.evaluate(args[i], ...range);
	});
	return yVals;
}

setType(type) {
	this.type = type;
	this.initializeInterpolators();
}

setCustomInterpolator(interpolator) {
	this.customInterpolator = interpolator;
	this.initializeInterpolators();
}

	  createDenseMap(input = null, type = 'linear', step = 0.005, evaluateValues = null) {
		  let xValues, yValues;
		  if (input == null) {
			  //for(let i = 0; i<this.x.length; i++){
				  ////////////////////////////////////////console.log(this.x[i]);
			//  }
			  type = this.type;
		    xValues = this.x;
		    yValues = this.y;
		  } else if (Array.isArray(input) && input.length > 0 && input[0] instanceof Point) {
		    xValues = input.map(point => point.coordinates.x);
		    yValues = input.map(point => point.coordinates.y);
		  } else if (Array.isArray(input) && input.length > 1) {
		    xValues = input[0];
		    yValues = input[1];
		  } else {
		    throw new Error('Invalid input format.');
		  }

		  let xToEvaluate = xValues;
		  if (evaluateValues) {
		    xToEvaluate = evaluateValues;
		  } else if (step) {
		    const xMin = Math.min(...xValues);
		    const xMax = Math.max(...xValues);
		    const numPoints = Math.floor((xMax - xMin) / step) + 1;
		    const stepPoints = Array.from({ length: numPoints }, (_, i) => xMin + i * step);

		    const primes = [2, 3, 5, 7, 11, 13, 17, 19];
		    const primePoints = [];
		    primes.forEach(prime => {
		      for (let i = 1; i <= prime-1; i++) {
		        const point = xMin + (i * (xMax-xMin) / prime);
		        if (point <= xMax) {
		          primePoints.push(point);
		        }
		      }
		    });
		    
		    xToEvaluate = [...stepPoints, ...primePoints];
		    xToEvaluate = Array.from(new Set(xToEvaluate)); // remove duplicates
		    xToEvaluate.sort((a, b) => parseFloat(a) - parseFloat(b)); // sort in ascending order
		  }

		  const interpolator = new Interpolator(xValues, yValues, type);
		  const denseMap = {};
		  xToEvaluate.forEach(x => {
		    denseMap[x] = Math.round(interpolator.evaluate(x) * 100) / 100;
		  });

		  // Sort keys in ascending order
		  const sortedKeys = Object.keys(denseMap).sort((a, b) => parseFloat(a) - parseFloat(b));

		  // Create new object with sorted keys
		  const sortedDenseMap = {};
		  sortedKeys.forEach(key => {
		    sortedDenseMap[key] = denseMap[key];
		  });

		  return sortedDenseMap;
		}

		  setData(data) {
			  const convertedData = MathematicalOpereration.convertDataStructure(data, 'points');
			  const x = convertedData.map(p => p.coordinates.x);
			  const y = convertedData.map(p => p.coordinates.y);
			  this.resetInterpolators(x, y);
			}
	derivative_at_x(x, h = 1e-5) {
		  return MathematicalOperation.derivative(x, this.evaluate.bind(this), h);
		}

	integral(a, b, n = 100) {

		  return MathematicalOperation.integral(a, b, n, this.evaluate.bind(this));
		}
	getDurationPoints(points) {
		  const intervals = [];
		  let currentPower = points[0].coordinates.y;
		  let currentInterval = new Interval(0, points[0].coordinates.x, new Point({ x: 0, y: currentPower }));
		  intervals.push(currentInterval);

		  for (let [point, nextPoint] of MathematicalOperation.zip(points, points.slice(1))) {
		    const targetDuration = nextPoint.coordinates.x - currentInterval.end;
		    const targetPower = nextPoint.coordinates.y;

		    if (targetDuration <= 0) {
		      throw new Error('Target duration must be greater than 0');
		    }

		    const averagePower = (targetPower + currentPower) / 2;
		    const targetEnergy = targetDuration * targetPower;
		    const currentEnergy = (currentInterval.end - currentInterval.start) * currentPower;
		    const newEnergy = targetEnergy - currentEnergy;
		    const newDuration = newEnergy / averagePower;

		    const newInterval = new Interval(currentInterval.end, currentInterval.end + newDuration, new Point({ x: newDuration, y: targetPower }));
		    intervals.push(newInterval);

		    currentPower = targetPower;
		    currentInterval = newInterval;
		  }

		  return intervals.map(interval => interval.point);
		}

	}


class PolynomialInterpolator {
	constructor( degree = null, domain, ...ranges) {
		if (ranges.length === 0) {
			throw new Error('At least one range is required');
		}
		this.domain = domain;
		this.ranges = ranges;
		this.degree = degree || this.ranges[0].length - 1;
		this.coeffs = this.computeCoeffs();
	}

	computeCoeffs() {
		let n = this.domain.length;
		let a = [];
		for (let i = 0; i < n; i++) {
			a[i] = [];
			for (let j = 0; j <= this.degree; j++) {
				a[i][j] = Math.pow(this.domain[i], j);
			}
			a[i][this.degree + 1] = this.ranges.map(range => range[i]);
		}

		for (let i = 0; i <= this.degree; i++) {
			for (let j = i + 1; j <= this.degree; j++) {
				for (let k = 0; k < n; k++) {
					let term = a[k][i];
					let denom = a[k][i-1] || 1;
					a[k][j] -= term * a[k][j-1] / denom;
				}
			}
		}

		let coeffs = [];
		for (let i = 0; i <= this.degree; i++) {
			let values = a.map(row => row[i]);
			let numerator = a.map(row => row[this.degree + 1]).reduce((acc, val, index) => acc + val * values[index], 0);
			let denominator = a.map(row => row[i] * row[i]).reduce((acc, val) => acc + val, 0);
			coeffs[i] = numerator / denominator;
		}

		return coeffs;
	}

	evaluate(x) {
		let y = 0;
		for (let i = 0; i <= this.degree; i++) {
			y += this.coeffs[i] * Math.pow(x, i);
		}
		return y;
	}
}
class BezierInterpolator {
    // Constructor method: Initializes the class with domain and ranges values
    constructor(domain, ranges) {
        this.domain = domain;
        this.ranges = ranges; //.length === 1 ? ranges[0] : ranges;
        this.degree = 17;//3;
        this.controlPoints = this.computeControlPoints();
    }

    // Method: computeControlPoints
    // Description: Calculates the control points of the Bezier curve using the De Casteljau algorithm
computeControlPoints() {
  let controlPoints = [];
  for (let i = 0; i < this.domain.length - 1; i++) {
    let p0 = [this.domain[i], ...this.ranges.map(range => range[i])];
    let p3 = [this.domain[i + 1], ...this.ranges.map(range => range[i + 1])];
    let dx = (this.domain[i + 1] - this.domain[i]) / 3;
    let p1 = [this.domain[i] + dx, ...this.ranges.map(range => range[i])];
    let p2 = [this.domain[i + 1] - dx, ...this.ranges.map(range => range[i + 1])];
    controlPoints.push([p0, p1, p2, p3]);
  }
  return controlPoints;
}

    // Method: evaluate
    // Description: Calculates the y-value at a given x-value using the Bezier curve
evaluate(x) {
  let results = [];
  for (let i = 0; i < this.domain.length - 1; i++) {
    if (x >= this.domain[i] && x <= this.domain[i + 1]) {
      let t = (x - this.domain[i]) / (this.domain[i + 1] - this.domain[i]);
      let [p0, p1, p2, p3] = this.controlPoints[i];
      let ti1 = 1 - t;
      let ti2 = ti1 * ti1;
      let ti3 = ti2 * ti1;
      let ti_1 = t;
      let ti_2 = ti_1 * ti_1;
      let ti_3 = ti_2 * ti_1;
      let xCoord = ti3 * p0[0] + 3 * ti2 * t * p1[0] + 3 * ti1 * ti_1 * p2[0] + ti_3 * p3[0];
      let yCoords = this.ranges.map((range, index) => ti3 * p0[index + 1] + 3 * ti2 * t * p1[index + 1] + 3 * ti1 * ti_1 * p2[index + 1] + ti_3 * p3[index + 1]);
      results = yCoords;
      break;
    }
  }
  return results;
}
}
class CatmullRomInterpolation {
  constructor(domain, ranges) {
    this.domain = domain;
    this.ranges = ranges;
    this.n = domain.length;
    this.tension = 0.5;
    this.computeSpline();
  }

  computeSpline() {
    this.b = [];
    for (let i = 0; i < this.n; i++) {
      this.b[i] = [];
      for (let j = 0; j < 4; j++) {
        let ti = this.getT(i, j);
        this.b[i][j] = this.getCatmullRomCoeff(ti);
      }
    }
  }

  getT(i, j) {
    if (j === 0) return this.domain[i];
    if (j === 1) return (this.domain[i] + this.domain[i+1]) / 2;
    if (j === 2) return this.domain[i+1];
    return (this.tension / (this.tension + 1)) * this.getT(i, j-1) + (1 / (this.tension + 1)) * this.getT(i+1, j-1);
  }

  getCatmullRomCoeff(t) {
    let t2 = t * t;
    let t3 = t2 * t;
    return 0.5 * (-t3 + 2 * t2 - t) + 0 * (t3 - 2 * t2 + 1) + 0.5 * (t3 - t2);
  }

  evaluate(x) {
    let result = [];
    for (let i = 0; i < this.ranges.length; i++) {
      let range = this.ranges[i];
      let y = 0;
      for (let j = 0; j < this.n-1; j++) {
        if (x >= this.domain[j] && x <= this.domain[j+1]) {
          let t = (x - this.domain[j]) / (this.domain[j+1] - this.domain[j]);
          let ti1 = 1 - t;
          let ti2 = ti1 * ti1;
          let ti3 = ti2 * ti1;
          let ti_1 = t;
          let ti_2 = ti_1 * ti_1;
          let ti_3 = ti_2 * ti_1;
          y = this.b[j][0] * ti_3 + this.b[j][1] * ti_2 * ti_1 + this.b[j][2] * ti_1 * ti_2 + this.b[j][3] * ti3;
          result.push(range[j] * ti_3 + range[j+1] * ti3 + 3 * range[j] * ti_2 * t + 3 * range[j+1] * ti_2 * ti_1);
          break;
        }
      }
    }
    return result;
  }
}
class PchipInterpolation {
  constructor(x, ys, tension = 1) {
    this.x = x;
    this.ys = ys;
    ////////////////////console.error("PCHIP INITIALIZATION",this.x, this.ys);
    this.n = x.length;
    this.m = Array(this.n-1).fill(0);
    this.tension = tension;
    this.computeSlopes();
  }

  computeSlopes() {
    for (let j = 0; j < this.ys.length; j++) {
      let y = this.ys[j];
      for (let i = 0; i < this.n-1; i++) {
        let delta = (y[i+1] - y[i]) / (this.x[i+1] - this.x[i]);
        let sign = delta > 0 ? 1 : -1;
        let deltam = this.m[i] * sign;
        if (delta == 0) {
          this.m[i] = 0;
          this.m[i+1] = 0;
        } else {
          let s = sign * Math.min(Math.abs(deltam), this.tension * Math.abs(delta));
          this.m[i] = s;
          this.m[i+1] = s;
        }
      }
    }
  }

  evaluate(x) {
    let results = Array(this.ys.length).fill(0);
    for (let j = 0; j < this.ys.length; j++) {
      let y = this.ys[j];
      let i = 0;
      while (x > this.x[i+1] && i < this.x.length-1) {
        i++;
      }
      let t = (x - this.x[i]) / (this.x[i+1] - this.x[i]);
      
      let h00 = 2 * t**3 - 3 * t**2 + 1;
      let h10 = t**3 - 2 * t**2 + t;
      let h01 = -2 * t**3 + 3 * t**2;
      let h11 = t**3 - t**2;
      ////////////////////console.error(t, h00, h10, h01, h11, y[i]);
      results[j] = h00 * y[i] + h10 * (this.x[i+1] - this.x[i]) * this.m[i] + h01 * y[i+1] + h11 * (this.x[i+1] - this.x[i]) * this.m[i+1];
    }
    ////////////////////console.log(results);
    return results;
  }
}
class CubicSplineInterpolation {
  constructor(domain, ranges) {
    if (!Array.isArray(domain) || domain.length < 2) {
      throw new Error('Invalid domain array');
    }
    if (ranges.length < 1) {
      throw new Error('At least one range array is required');
    }
    if (!ranges.every(r => Array.isArray(r) && r.length === domain.length)) {
      throw new Error('Invalid range array(s)');
    }
    this.domain = domain;
    this.ranges = ranges;
    this.n = domain.length;
    this.a = [];
    this.b = [];
    this.c = [];
    this.d  = [];
    ////////////////console.warn(domain, ranges, this.n);
    this.computeCoefficients();
  }

  computeCoefficients() {
    let h = Array(this.n - 1).fill(0);
    let alphas = [];
    for (let j = 0; j < this.ranges.length; j++) {
      let alpha = Array(this.n - 1).fill(0);
      for (let i = 0; i < this.n - 1; i++) {
        h[i] = this.domain[i + 1] - this.domain[i];
        alpha[i] = 3 * (this.ranges[j][i + 1] - this.ranges[j][i]) / h[i] - 3 * (this.ranges[j][i] - (i === 0 ? this.ranges[j][0] : this.ranges[j][i - 1])) / (i === 0 ? h[0]:h[i - 1]);
      }
      alphas.push(alpha);
    }
////////////////console.warn("DOMAIN",this.domain,"RANGES", this.ranges,"H", h, "ALPHAS",alphas);

    let l = Array(this.n).fill(0);
    let mu = Array(this.n - 1).fill(0);
    let z = Array(this.n).fill(0);
    let coefficients = [];

    for (let j = 0; j < this.ranges.length; j++) {
      let alpha = alphas[j];
      l[0] = 1;
      mu[0] = 0;
      z[0] = 0;

      for (let i = 1; i < this.n - 1; i++) {
        let factor = 1 / (4 - mu[i - 1]);
        mu[i] = factor;
        l[i] = 2 - factor * h[i - 1];
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i];
      }

      let b = Array(this.n).fill(0);
      let c = Array(this.n + 1).fill(0);
      let d = Array(this.n).fill(0);

      l[this.n - 1] = 1;
      z[this.n - 1] = 0;
      c[this.n] = 0;

      for (let i = this.n - 2; i >= 0; i--) {
        c[i + 1] = z[i] - mu[i] * c[i + 2];
        b[i + 1] = (this.ranges[j][i + 1] - this.ranges[j][i]) / h[i] - h[i] * (c[i + 2] + 2 * c[i + 1]) / 3;
        d[i + 1] = (c[i + 2] - c[i + 1]) / (3 * h[i]);
      }

  this.a.push(this.ranges[j]);
  this.b.push(b);
  this.c.push(c);
  this.d.push(d);
    }
}

evaluate(x) {
let results = [];
for (let j = 0; j < this.ranges.length; j++) {
let i = this.getInterval(x, j);
let dx = x - this.domain[i];
let value = this.a[j][i] + this.b[j][i] * dx + this.c[j][i] * dx ** 2 + this.d[j][i] * dx ** 3;
////////////////console.warn(value, this.a, this.b, dx, this.c, this.d);
results.push(value);
}
return results;
}

getInterval(x, j) {
let i = 0;
if (x >= this.domain[this.n - 1]) {
i = this.n - 2;
} else {
while (x > this.domain[i + 1]) {
i++;
}
}
return i;
}
}
class BSplineInterpolation {
  constructor(x, ...ys) {
    this.x = x;
    this.ys = ys;
    this.n = x.length;
    this.computeSpline();
  }

  computeSpline() {
    let t = [];
    for (let i = 0; i < this.n; i++) {
      t[i] = i;
    }

    let u = [];
    for (let i = 0; i <= this.n + this.k; i++) {
      u[i] = i;
    }

    this.b = [];
    for (let i = 0; i < this.n; i++) {
      this.b[i] = [];
      for (let j = 0; j < this.k + 1; j++) {
        this.b[i][j] = 0;
        for (let l = i; l < i + this.k + 1; l++) {
          this.b[i][j] +=
            this.basisFunction(j, l, t, u) *
            this.ys.map((y) => y[l]);
        }
      }
    }
  }

  basisFunction(j, l, t, u) {
    if (j == 0) {
      if (t[l] <= u[j] && u[j + 1] <= t[l + 1]) {
        return 1;
      } else {
        return 0;
      }
    } else {
      let a =
        u[j + j] - u[j] == 0
          ? 0
          : ((u[j + j] - u[j]) *
              this.basisFunction(j - 1, l, t, u)) /
            (u[j + j] - u[j]);
      let b =
        u[j + j + 1] - u[j + 1] == 0
          ? 0
          : ((u[j + j + 1] - u[j + 1]) *
              this.basisFunction(j - 1, l + 1, t, u)) /
            (u[j + j + 1] - u[j + 1]);
      return a + b;
    }
  }

  evaluate(x, k = 2) {
    let result = Array(this.ys.length).fill(0);
    for (let i = 0; i < this.n; i++) {
      if (x >= this.x[i] && x <= this.x[i + 1]) {
        let t = (x - this.x[i]) / (this.x[i + 1] - this.x[i]);
        for (let j = 0; j < k + 1; j++) {
          result = result.map((res, idx) => {
            return (
              res +
              this.b[i][j] *
                Math.pow(t, j) *
                Math.pow(1 - t, k - j) *
                this.ys[idx][i]
            );
          });
        }
        break;
      }
    }
    return result;
  }
}
class Wavelet {
	  constructor(coefficients) {
	    this.coefficients = coefficients;
	    this.numLevels = Math.log2(coefficients.length);
	    this.lowPassFilter = [0.7071067811865476, 0.7071067811865476];
	    this.highPassFilter = [-0.7071067811865476, 0.7071067811865476];
	  }

	  forwardTransform() {
	    let temp = [];
	    for (let i = 0; i < this.coefficients.length; i++) {
	      temp.push(this.coefficients[i]);
	    }

	    for (let i = 0; i < this.numLevels; i++) {
	      let half = temp.length / 2;
	      let lowPass = [];
	      let highPass = [];
	      for (let j = 0; j < half; j++) {
	        lowPass.push(0);
	        highPass.push(0);
	        for (let k = 0; k < 2; k++) {
	          let index = ((2 * j + k) % temp.length);
	          lowPass[j] += temp[index] * this.lowPassFilter[k];
	          highPass[j] += temp[index] * this.highPassFilter[k];
	        }
	      }
	      temp = lowPass;
	      for (let j = 0; j < half; j++) {
	        temp.push(highPass[j]);
	      }
	    }
	    return temp;
	  }

	  inverseTransform() {
	    let temp = [];
	    for (let i = 0; i < this.coefficients.length; i++) {
	      temp.push(this.coefficients[i]);
	    }

	    for (let i = this.numLevels - 1; i >= 0; i--) {
	      let half = temp.length / 2;
	      let lowPass = [];
	      let highPass = [];
	      for (let j = 0; j < half; j++) {
	        lowPass.push(0);
	        highPass.push(0);
	        for (let k = 0; k < 2; k++) {
	          let index = ((j - this.coefficients.length / 2 + k + temp.length) % temp.length);
	          lowPass[j] += temp[index] * this.lowPassFilter[k];
	          highPass[j] += temp[index + half] * this.highPassFilter[k];
	        }
	      }
	      temp = [];
	      for (let j = 0; j < half; j++) {
	        temp.push(lowPass[j] + highPass[j]);
	      }
	      for (let j = 0; j < half; j++) {
	        temp.push(lowPass[j] - highPass[j]);
	      }
	    }
	    return temp;
	  }
	}
const typeWavelets = {
		  
		  db1: [0.7071067811865476, 0.7071067811865476],
		  db2: [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604],
		  db3: [0.3326705529500825, 0.8068915093110928, 0.4598775021184915, -0.13501102001039084, -0.0854412738820267, 0.0352262918857096],
		  db4: [0.2303778133088965, 0.7148465705529154, 0.6308807679295904, -0.02798376941698385, -0.18703481171888114, 0.030841381835986965, 0.03288301166688522, -0.010597401785069482],
		  db5: [0.160102397974125, 0.6038292697974729, 0.7243085284377729, 0.13842814590132035, -0.24229488706638238, -0.03224486958502952, 0.07757149384006515, -0.006241490212798291, -0.012580751999081991, 0.0033357252854737713],
		  coif1: [0.038580777748, -0.126969125396, -0.077161555496, 0.607491641386, 0.745687558934, 0.226584265197, -0.129766867567, 0.093310490563],
		  coif2: [0.016387336463, -0.041464936782, -0.067372554722, 0.386110066823, 0.812723635450, 0.417005184424, -0.076488599078, -0.059434418646, 0.023680171947, 0.005611434819],
		  coif3: [-0.003793512864, 0.007782596426, 0.023452696142, -0.065771911281, -0.061123390003, 0.405176902410, 0.793777222626, 0.428483476377, -0.071799821619, -0.082301927106, 0.034555027573, 0.015880544864],
		  coif4: [-0.000892313668, 0.001629492013, 0.007346166328, -0.023680171947, -0.059434418646, 0.293654040046, 0.767556669298, 0.536101917090, 0.017441255087, -0.111540743350, 0.040939156701, 0.010774605374, -0.002228417236, -0.001075880228], 
		  coif5: [0.000553842201, -0.001662863702, -0.007761464640, 0.018073643235, 0.040823227229, -0.141996101046, -0.194450471766, 0.109441219470, 0.484572603419, 0.511819418570, 0.097609697812, -0.031257111869, 0.012090669864, 0.003490712084, -0.000771632055],
		  sym1:  [   -0.1294095226,  0.2241438680,  0.8365163037,  0.4829629131, -0.0495528349, -0.2312317420,   0.3152503517,  0.7511339080, 0.4946238904, -0.1115407433, -0.2673164462, 0.0416305449, 0.0409688106, -0.0249247748, -0.0087460940,  0.0048703530 ],
sym2: [-0.129409522551, 0.224143868042, 0.836516303738, 0.482962913145],
sym3: [0.035226291886, -0.085441273882, -0.135011020010, 0.459877502118, 0.806891509311, 0.332670552950, -0.094282794109, -0.027152459411, 0.016574541631]
, sym4: [-0.075765714789, -0.029635527646, 0.497618667638, 0.803738751805, 0.297857795610, -0.099219543576, -0.012603967262, 0.032223100604, -0.010597401785, -0.063423780966, 0.030224878858, 0.006594748445]
, sym5: [0.027333068345, 0.029519490926, -0.039134249302, 0.199397533976, 0.723407690919, 0.633978963458, 0.016602105764, -0.175328089908, -0.021101834026, 0.019538882735, -0.009007976137, -0.006840316244, 0.002732376773, 0.000428394300, -0.000207123219]
, bior11: [0.707106781187, 0.707106781187, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
, bior13: [0.066291260736, -0.198873782224, -0.154679608384, 0.994368911043, 0.994368911043, -0.154679608384, -0.198873782224, 0.066291260736, 0.0, 0.0]
, bior15: [-0.001077301085, 0.004777257511, 0.000553842201, -0.031582039318, 0.027522865530, 0.097501605587, -0.129766867567, -0.226264693965, 0.315250351709, 0.751133908021, 0.494623890398, -0.111540743350, -0.054750583811, 0.017441255087, 0.011936235524, -0.003976621945, -0.001077301085]
};
class WaveletInterpolation {
  constructor(x, ys, wavelet) {
    this.x = x;
    this.ys = ys;
    this.wavelet = wavelet;
    this.interpolations = this.buildInterpolations(x, ys, wavelet);
  }

  // Builds an interpolation for each dimension and returns them in an object
  buildInterpolations(x, ys, wavelet) {
    const interpolations = {};
    for (let i = 0; i < ys.length; i++) {
      const y = ys[i];
      const interpolation = this.buildInterpolation(x, y, wavelet);
      interpolations[i] = interpolation;
    }
    return interpolations;
  }

  // Builds an interpolation for a given set of x and y values
  buildInterpolation(x, y, wavelet) {
    // Create a new interpolation instance
    const interpolation = new Interpolator(x, y);

    // Add the wavelet interpolation functions to the interpolation instance
    interpolation.evaluateWavelet = this.evaluateWavelet.bind(this);
    interpolation.slopeWavelet = this.slopeWavelet.bind(this);

    // Return the interpolation instance
    return interpolation;
  }

  // Evaluates a wavelet at a given point x
  evaluateWavelet(x, wavelet) {
    let y = 0;
    const waveletCoefficients = this.wavelet[wavelet];
    for (let i = 0; i < waveletCoefficients.length; i++) {
      y += waveletCoefficients[i] * this.basisFunction(x, i);
    }
    return y;
  }

  // Evaluates the slope of a wavelet at a given point x
  slopeWavelet(x, wavelet) {
    let m = 0;
    const waveletCoefficients = this.wavelet[wavelet];
    for (let i = 0; i < waveletCoefficients.length; i++) {
      m += waveletCoefficients[i] * this.slopeBasisFunction(x, i);
    }
    return m;
  }

  // Calculates the value of the wavelet basis function at a given point
  basisFunction(x, i) {
    let result = 1;
    if (i % 2 == 0) {
      result = result * Math.cos(x * Math.PI * (i + 1) / 2);
    } else {
      result = result * Math.sin(x * Math.PI * (i + 1) / 2);
    }
    return result;
  }

  // Calculates the slope of the wavelet basis function at a given point
  slopeBasisFunction(x, i) {
    let result = 0;
    if (i % 2 == 0) {
      result = -1 * Math.PI * (i + 1) / 2 * Math.sin(x * Math.PI * (i + 1) / 2);
    } else {
      result = Math.PI * (i + 1) / 2 * Math.cos(x * Math.PI * (i + 1) / 2);
    }
    return result;
  }

  // Evaluates the interpolation at a given point x
  evaluate(x) {
    const ys = this.ys;
    const interpolations = this.interpolations;
    const y = [];
    for (let i = 0; i < ys.length; i++) {
      y.push(interpolations[i].evaluate(x));
    }
    return y;
  }

  // Evaluates the

// derivative of the interpolation at a given point x
evaluateDerivative(x) {
const ys = this.ys;
const interpolations = this.interpolations;
const m = [];
for (let i = 0; i < ys.length; i++) {
m.push(interpolations[i].slope(x));
}
return m;
}
}

class AkimaInterpolation {
  constructor(domain, ranges, coeff = null) {
    this.x = domain;
    this.y = ranges;
    this.coeff = coeff ? coeff : 0;
    this.m = domain.length;
    this.coeffs = this.computeCoeffs();
  }

  computeCoeffs() {
    const m = this.m;
    const h = new Array(m - 1);
    for (let i = 0; i < m - 1; i++) {
      h[i] = this.x[i + 1] - this.x[i];
    }
    const mu = new Array(m - 2);
    const z = new Array(m - 2);
    for (let i = 1; i < m - 1; i++) {
      mu[i - 1] = h[i - 1] / (h[i - 1] + h[i]);
      z[i - 1] = 3 * ((this.y[i + 1][0] - this.y[i][0]) / h[i] - (this.y[i][0] - this.y[i - 1][0]) / h[i - 1]);
    }
    const l = new Array(m - 2);
    const u = new Array(m - 2);
    const d = new Array(m - 2);
    for (let i = 1; i < m - 2; i++) {
      l[i] = mu[i];
      u[i] = 1 - l[i];
      d[i] = (z[i] - l[i] * z[i - 1]) / (u[i] * d[i - 1] + 2);
    }
    const c = new Array(m - 1);
    c[m - 2] = 0;
    for (let i = m - 3; i >= 0; i--) {
      c[i] = d[i] - u[i] * c[i + 1];
    }
    const aCoeffs = new Array(m - 1);
    const bCoeffs = new Array(m - 1);
    const dCoeffs = new Array(m - 1);
    for (let i = 0; i < m - 1; i++) {
      aCoeffs[i] = this.y[i][0];
      if (i < m - 2) {
        bCoeffs[i] = (this.y[i + 1][0] - this.y[i][0]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3;
        dCoeffs[i] = (c[i + 1] - c[i]) / (3 * h[i]);
      }
    }
    const coeffs = new Array(m - 1);
    for (let i = 0; i < m - 1; i++) {
      coeffs[i] = {
        a: aCoeffs[i],
        b: bCoeffs[i],
        c: c[i],
        d: dCoeffs[i],
        x1: this.x[i],
        x2: this.x[i + 1]
      };
    }
    return coeffs;
  }


evaluate(x) {
    let coeffs = this.coeffs;
    let n = this.m - 1;
    let values = new Array(n);
    for (let j = 0; j < n; j++) {
      let i = j;
      if (x < this.x[i]) {
        i = 0;
      } else if (x >= this.x[n - 1]) {
        i = n - 2;
      } else {
        while (x < this.x[i] || x >= this.x[i + 1]) {
          i++;
        }
      }
      let t = (x - this.x[i]) / (this.x[i + 1] - this.x[i]);
      values[j] = coeffs[i].a + t * (coeffs[i].b + t * (coeffs[i].c + t * coeffs[i].d));
    }
    return this.coeff === null ? values[0] : values;
  }
  }
  class LinearInterpolation {
  constructor(domain, ranges) {
	  //////////////////console.error(domain, ranges);
    if (!Array.isArray(domain) || domain.length < 2) {
      throw new Error('Invalid domain array');
    }
    if (ranges.length < 1) {
      throw new Error('At least one range array is required');
    }
    if (!ranges.every(r => Array.isArray(r) && r.length === domain.length)) {
      throw new Error('Invalid range array(s)');
    }
    this.domain = domain;
    this.ranges = ranges;
  }

  evaluateAxis(x, yValues, options = {}) {
    const {
      method = 'nearest',
      extrapolate = true,
      defaultValue = NaN
    } = options;

    let i = 0;
    while (i < this.domain.length - 1 && x > this.domain[i + 1]) {
      i++;
    }

    if (i === 0 && !extrapolate) {
      return defaultValue;
    }
    if (i === this.domain.length - 2 && !extrapolate) {
      return defaultValue;
    }

    let x1 = this.domain[i], x2 = this.domain[i + 1];
    let y1 = yValues[i], y2 = yValues[i + 1];

    if (method === 'linear') {
      y1 = this.interpolateLinear(x1, y1, x2, y2, x);
    } else if (method === 'nearest') {
      y1 = this.interpolateNearest(x1, y1, x2, y2, x);
    }

    return y1;
  }

  interpolateLinear(x1, y1, x2, y2, x) {
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
  }

  interpolateNearest(x1, y1, x2, y2, x) {
    if (x - x1 < x2 - x) {
      return y1;
    } else {
      return y2;
    }
  }

  evaluate(x, options = {}) {
    const {
      method = 'nearest',
      extrapolate = true,
      defaultValue = NaN
    } = options;

    const yValues = this.ranges.map(r => this.evaluateAxis(x, r, options));
    return yValues;
  }
}

class VectorInterpolator extends Interpolator {
  constructor(domain, ...ranges) {
    super(domain, ...ranges);
    this.uValues = null;
    this.vValues = null;
  }
calculateW( x, y, z, u, v ) {
  

  // Calculate the derivative of u and v with respect to t
  const du_dt = this.uValues ? this.uValues.derivative().evaluate(u) : 0;
  const dv_dt = this.vValues ? this.vValues.derivative().evaluate(v) : 0;

  // Calculate the magnitude of the acceleration vector
  const a = Math.sqrt(Math.pow(du_dt, 2) + Math.pow(dv_dt, 2));

  // Calculate the cross product of the velocity vector and acceleration vector
  const crossProduct = [
    u * dv_dt - v * du_dt,
    v * a,
    -u * a
  ];

  // Calculate the dot product of the velocity vector and acceleration vector
  const dotProduct = du_dt * x + dv_dt * y;

  // Calculate the impulse
  const w = dotProduct * z + crossProduct[0] * crossProduct[1] + crossProduct[2] * crossProduct[1];

  return w;
}
  evaluateND(...args) {
    const t = args[0];
    const { x, y, z } = super.evaluateND(...args);
    const u = this.uValues ? this.uValues.evaluate(t) : null;
    const v = this.vValues ? this.vValues.evaluate(t) : null;
    const w = this.calculateW(x, y, z, u, v);
    
    return { x, y, z, u, v, w };
  }

  getVectorPoints(numPoints) {
    const domain = this.domain;
    const stepSize = (domain[domain.length - 1] - domain[0]) / (numPoints - 1);

    const points = [];
    for (let i = 0; i < numPoints; i++) {
      const t = domain[0] + i * stepSize;
      const { x, y, z, u, v, w } = this.evaluate(t);
      points.push(new Point({ x, y, z, t, u, v, w }));
    }

    return points;
  }

  getRangeEvaluator(index, param) {
    const range = this.ranges[index];
    const interpolator = this.getSubInterpolator(range);

    return (val) => {
      if (param === 't') {
        return interpolator.evaluate(val);
      } else if (param === 'u' || param === 'v') {
        const axisIndex = (param === 'u') ? 0 : 1;
        const axisValues = range.slice(axisIndex, axisIndex + 3);
        const diffValues = axisValues.map((v, i) => axisValues[i+1] - v).slice(0, -1);
        const diffSum = diffValues.reduce((sum, v) => sum + v, 0);
        const t = this.findTFromUorV(val, diffValues, diffSum);
        const subInterpolator = this.getSubInterpolator(axisValues);
        return subInterpolator.evaluate(t);
      }
    };
  }

  findTFromUorV(val, diffValues, diffSum) {
    const n = diffValues.length + 1;
    const A = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    const B = Array.from({ length: n }, () => 0);
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          if (i === 0) {
            A[i][j] = 1;
          } else if (i === n - 1) {
            A[i][j] = 1;
          } else {
            A[i][j] = 2;
          }
        } else if (i === j - 1) {
          A[i][j] = 1;
        } else if (i === j + 1) {
          A[i][j] = 1;
        }
      }
      if (i === 0) {
        B[i] = 0;
      } else if (i === n - 1) {
        B[i] = diffSum;
      } else {
        B[i] = 3 * (diffValues[i-1] / (domain[n-1] - domain[0]));
}
}


const M = MathUtil.solveLinearSystem(A, B);
const C = Array.from({ length: n }, (_, i) => {
  if (i === 0) {
    return 0;
  } else {
    return M[i-1];
  }
});

let k = 0;
while (k < n-1 && C[k+1] < val) {
  k++;
}

const t = domain[k] + (val - C[k]) / (3 * (diffValues[k] / (domain[n-1] - domain[0])));

return t;
}

setUValues(uValues) {
this.uValues = uValues;
}

setVValues(vValues) {
this.vValues = vValues;
}
}

class MathUtil {
static solveLinearSystem(A, B) {
const n = A.length;
const L = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
const U = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
const X = Array.from({ length: n }, () => 0);


for (let j = 0; j < n; j++) {
  U[0][j] = A[0][j];
  L[j][j] = 1;
}

for (let i = 1; i < n; i++) {
  L[i][0] = A[i][0] / U[0][0];
}

for (let i = 1; i < n; i++) {
  for (let j = i; j < n; j++) {
    let sum = 0;
    for (let k = 0; k < i; k++) {
      sum += L[i][k] * U[k][j];
    }
    U[i][j] = A[i][j] - sum;
  }
  for (let j = i+1; j < n; j++) {
    let sum = 0;
    for (let k = 0; k < i; k++) {
      sum += L[j][k] * U[k][i];
    }
    L[j][i] = (A[j][i] - sum) / U[i][i];
  }
}

for (let i = 0; i < n; i++) {
  let sum = 0;
  for (let j = 0; j < i; j++) {
    sum += L[i][j] * X[j];
  }
  X[i] = B[i] - sum;
}

for (let i = n-1; i >= 0; i--) {
  let sum = 0;
  for (let j = i+1; j < n; j++) {
    sum += U[i][j] * X[j];
  }
  X[i] = (X[i] - sum) / U[i][i];
}

return X;

}

getSubInterpolator(range) {
  const domain = this.domain.slice();
  const ranges = [range];

  // Find the start and end index for the range in the current domain
  const startIndex = this.findNearestIndex(domain, range[0]);
  const endIndex = this.findNearestIndex(domain, range[range.length - 1]);

  // Trim the domain and ranges to the sub-range
  domain.splice(endIndex, domain.length - endIndex);
  domain.splice(0, startIndex);
  for (let i = 0; i < ranges.length; i++) {
    ranges[i].splice(endIndex, ranges[i].length - endIndex);
    ranges[i].splice(0, startIndex);
  }

  // Create a new interpolator for the sub-range
  return new LinearInterpolator(domain, ...ranges);
}

findNearestIndex(arr, val) {
  return arr.reduce((prev, curr, index) => {
    return Math.abs(curr - val) < Math.abs(arr[prev] - val) ? index : prev;
  }, 0);
}
}