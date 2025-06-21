class Graph {
  constructor(canvas, dataTextArea) {
    this.dataTextArea = dataTextArea;
    this.onScreenCanvas = canvas;
    this.canvas = document.createElement('canvas');
    this.canvas.width = canvas.width;
    this.canvas.height = canvas.height;
    this.paddingX = 2;
    this.paddingY = 4;
    this.onScreenContext = this.onScreenCanvas.getContext('2d');
    this.context = this.canvas.getContext('2d');
    this.origin = { x: 0, y: this.canvas.height }; // default origin is bottom-left corner
    this.animationFrameID = null;
    this.hasChanged = false;
    this.startUpdateCanvas();
  }

  startUpdateCanvas() {
    const updateCanvas = () => {
      if (this.hasChanged) {
        this.onScreenContext.clearRect(0, 0, this.onScreenCanvas.width, this.onScreenCanvas.height);
        this.onScreenContext.drawImage(this.canvas, 0, 0);
        this.hasChanged = false;
      }
      this.animationFrameID = requestAnimationFrame(updateCanvas);
    };

    this.animationFrameID = requestAnimationFrame(updateCanvas);
  }
attachToElement(element) {
  element.appendChild(this.canvas);


}
		  setOrigin(x, y) {
		    this.origin = { x: x, y: this.canvas.height - y };
		  }
		  formatPolarData(points) {
  ////////////////console.warn("DRAWING A POLAR GRAPH");
  const formattedData = {
    r: [],
    theta: [],
    keys: []
  };

  const maxRadius = Math.max(...points.map(point => point.coordinates.r));
  const tickCount = 10;
  const xPadding = this.paddingX;
  const yPadding = this.paddingY;

  this.setOrigin(this.canvas.width / 2, this.canvas.height /2 );
  
  let yScale = ((this.canvas.height-yPadding)/ parseFloat(2))/maxRadius;
  
  this.scale = {x: 1, y: yScale};

  this.drawPolarAxes(maxRadius, tickCount);

  points.forEach(point => {
    let r;
    let theta;
    if (point.coordinates.r && point.coordinates.theta) {
      r = point.coordinates.r;
      theta = point.coordinates.theta;
    } else {
      r = point.coordinates.y;
      theta = point.coordinates.x;
    }
    if (r < 0) {
      r = -r;
      theta = theta + Math.PI;
    }
    formattedData.r.push(r * yScale);
    formattedData.theta.push(theta);
    Object.keys(point.coordinates).forEach(key => {
      if (!formattedData.keys.includes(key)) {
        formattedData.keys.push(key);
      }
    });
  });

  return formattedData;
}

			drawPolarAxes(maxRadius, tickCount = 10) {
			  const tickSpacing = maxRadius / tickCount;

			  // draw radial lines
			  this.context.strokeStyle = '#ddd';
			  this.context.lineWidth = 0.5;
			  for (let i = 1; i <= tickCount; i++) {
			    const r = i * tickSpacing * this.scale.y;
			    this.context.beginPath();
			    this.context.arc(this.origin.x, this.origin.y, r, 0, 2 * Math.PI);
			    this.context.stroke();
			  }

			  // draw angle ticks
			  this.context.fillStyle = '#000';
			  this.context.font = '12px sans-serif';
			  this.context.textAlign = 'center';
			  for (let i = 0; i < 360; i += 30) {
			    const theta = i * Math.PI / 180;
			    const x = this.origin.x + maxRadius * this.scale.y * Math.cos(theta);
			    const y = this.origin.y - maxRadius * this.scale.y * Math.sin(theta);
			    this.context.beginPath();
			    this.context.moveTo(this.origin.x, this.origin.y);
			    this.context.lineTo(x, y);
			    this.context.stroke();
			    this.context.fillText(`${i}°`, x, y);
			  }

			  // reset stroke style and line width
			  this.context.strokeStyle = '#000';
			  this.context.lineWidth = 1;
			}
		  drawAxes(xLabel, yLabel, xTickCount = 10, yTickCount = 10) {
		    const xPadding = this.paddingX;
		    const yPadding = this.paddingY;
		    const xMax = this.canvas.width - xPadding;
		    const yMax = this.canvas.height - yPadding;

		    // draw x-axis
		    this.context.beginPath();
		    this.context.moveTo(this.origin.x + xPadding, this.origin.y);
		    this.context.lineTo(this.origin.x + xMax, this.origin.y);
		    this.context.stroke();

		    // draw y-axis
		    this.context.beginPath();
		    this.context.moveTo(this.origin.x + xPadding, this.origin.y);
		    this.context.lineTo(this.origin.x + xPadding, this.origin.y - yMax);
		    this.context.stroke();

		    // draw x-axis label
		    this.context.font = '12px sans-serif';
		    this.context.textAlign = 'center';
		    this.context.fillText(xLabel, this.origin.x + (xPadding + xMax) / 2, this.origin.y + 30);

		    // draw y-axis label
		    this.context.save();
		    this.context.translate(this.origin.x - 30, this.origin.y - (yPadding + yMax) / 2);
		    this.context.rotate(-Math.PI / 2);
		    this.context.textAlign = 'center';
		    this.context.fillText(yLabel, 0, 0);
		    this.context.restore();

		    // draw grid lines
		    this.context.strokeStyle = '#ddd';
		    this.context.lineWidth = 0.5;
		    for (let i = 1; i < xTickCount; i++) {
		      const x = this.origin.x + xPadding + (i / xTickCount) * (xMax - xPadding);
		      this.context.beginPath();
		      this.context.moveTo(x, this.origin.y - yPadding);
		      this.context.lineTo(x, this.origin.y - yMax);
		      this.context.stroke();
		    }
		    for (let i = 1; i < yTickCount; i++) {
		      const y = this.origin.y - yPadding - (i / yTickCount) * (yMax - yPadding);
		      this.context.beginPath();
		      this.context.moveTo(this.origin.x + xPadding, y);
		      this.context.lineTo(this.origin.x + xMax, y);
		      this.context.stroke();
		    }

		    // reset stroke style and line width
		    this.context.strokeStyle = '#000';
		    this.context.lineWidth = 1;
		  }
		  formatData(points, options) {
			  const formattedData = {
			    x: [],
			    y: [],
			    keys: []
			  };
//////////////console.warn(points);
			  const xMax = Math.max(...points.map(point => point.coordinates.x));
			  const xMin = Math.min(...points.map(point => point.coordinates.x));
			  const yMax = Math.max(...points.map(point => point.coordinates.y));
			  const yMin = Math.min(...points.map(point => point.coordinates.y));
			  
			    const yRange = yMax - yMin;
			    const xRange = xMax - xMin;

			    const xPadding = this.paddingX;
			    const yPadding = this.paddingY;

			    let xScale = (this.canvas.width - xPadding) / xRange * options.scaleZoom;
			    let yScale = (this.canvas.height - yPadding) / yRange * options.scaleZoom;
			    if(options.scaleMode != 'fit'){
			    if(yScale < xScale){
			    	xScale = yScale;
			    }else{
			    	yScale = xScale;
			    }
			    }
			    this.scale = {x:xScale, y:yScale}
			    
			    let xMid = (xMin + xMax) / parseFloat(2);
			    let yMid = (yMin + yMax) / parseFloat(2);
			    this.setOrigin(((this.canvas.width-xPadding) / 2) - xScale*xMid, ((this.canvas.height+yPadding) / 2) - yScale*yMid);
			    //////////////////////////////////////console.warn(this.origin);
			  
			  this.drawAxes("("+xMin+") Scale factor: "+xScale+" ("+xMax+")", "("+yMin+") Scale factor: "+yScale+" ("+yMax+")");
			  
			  points.forEach(point => {
			    const x = point.coordinates.x * xScale;
			    const y = point.coordinates.y * yScale;
			    formattedData.x.push(x+xPadding);
			    formattedData.y.push(y);
			    Object.keys(point.coordinates).forEach(key => {
			      if (!formattedData.keys.includes(key)) {
			        formattedData.keys.push(key);
			      }
			    });
			  });
////////////////////////////////////console.warn(formattedData);
			  return formattedData;
			}

drawLineChart(points, options, steps = 1000, domain = 'index', interpolationType = 'linear') {
  const coordType = options.coordType;
  const formattedData = coordType === 'Polar' ? this.formatPolarData(points) : this.formatData(points, options);
  const xValues = coordType === 'Polar' ? formattedData.r : formattedData.x;
  const yValues = coordType === 'Polar' ? formattedData.theta : formattedData.y;

  const pointCount = xValues.length;
  const segmentCount = pointCount - 1;
  const segmentLengths = [];
  let totalLength = 0;

  for (let i = 0; i < segmentCount; i++) {
    const x1 = coordType === 'Polar' ? yValues[i] * Math.cos(xValues[i]) : xValues[i];
    const y1 = coordType === 'Polar' ? yValues[i] * Math.sin(xValues[i]) : yValues[i];
    const x2 = coordType === 'Polar' ? yValues[i + 1] * Math.cos(xValues[i + 1]) : xValues[i + 1];
    const y2 = coordType === 'Polar' ? yValues[i + 1] * Math.sin(xValues[i + 1]) : yValues[i + 1];
    const segmentLength = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    segmentLengths.push(segmentLength);
    totalLength += segmentLength;
  }
  //console.error(points);
const dataset = new DataSet(points);
////////////console.warn(dataset);
  const colorFuncs = dataset.getColorFuncs(domain, interpolationType);

  for (let i = 0; i < segmentCount; i++) {
    const x1 = coordType === 'Polar' ? this.origin.x + (xValues[i] * Math.cos(yValues[i])) : this.origin.x + xValues[i];
    const y1 = coordType === 'Polar' ? this.origin.y - (xValues[i] * Math.sin(yValues[i])) : this.origin.y - yValues[i];
    const x2 = coordType === 'Polar' ? this.origin.x + (xValues[i + 1] * Math.cos(yValues[i + 1])) : this.origin.x + xValues[i + 1];
    const y2 = coordType === 'Polar' ? this.origin.y - (xValues[i + 1] * Math.sin(yValues[i + 1])) : this.origin.y - yValues[i + 1];

    const segmentSteps = Math.ceil(segmentLengths[i] / totalLength * steps);
    const segmentStepSize = 1 / segmentSteps;

    for (let j = 0; j < segmentSteps; j++) {
      const t = i + j * segmentStepSize;
      const x = x1 + (x2 - x1) * (t - i);
      const y = y1 + (y2 - y1) * (t - i);

      const rVal = colorFuncs['R'](t);
      const gVal = colorFuncs['G'](t);
      const bVal = colorFuncs['B'](t);
      const aVal = colorFuncs['A'](t);

this.context.beginPath();
this.context.strokeStyle = `rgba(${rVal},${gVal},${bVal},${aVal})`;
this.context.lineWidth = options.width; // Set line width to 2 pixels
this.context.moveTo(x1, y1);
this.context.lineTo(x2, y2);
this.context.stroke();
this.hasChanged = true;
    }
  }
}
	
			  drawBarChart(dataset, coordType) {
				  if (coordType === 'polar') {
				    const formattedData = this.formatPolarData(dataset);
				    const barCount = formattedData.r.length;
				    const barWidth = (2 * Math.PI) / barCount;

				    this.context.strokeStyle = '#000';
				    this.context.lineWidth = 3;
				    for (let i = 0; i < barCount; i++) {
				      const r = formattedData.r[i];
				      const theta = formattedData.theta[i];
				      const x = this.origin.x + r * Math.cos(theta);
				      const y = this.origin.y - r * Math.sin(theta);
				      const barLength = r;

				      this.context.beginPath();
				      this.context.moveTo(this.origin.x, this.origin.y);
				      this.context.lineTo(x, y);

				      const point = dataset.points[i];
				      let rVal, gVal, bVal, aVal;
				      if (point.coordinates.R != null && point.coordinates.G != null && point.coordinates.B != null) {
				        rVal = point.coordinates.R;
				        gVal = point.coordinates.G;
				        bVal = point.coordinates.B;
				        aVal = point.coordinates.A != null ? point.coordinates.A : 1.0; // default to fully opaque if alpha is not specified
				      } else {
				        rVal = 0;
				        gVal = 0;
				        bVal = 0;
				        aVal = 1.0;
				      }
				      this.context.fillStyle = `rgba(${rVal},${gVal},${bVal},${aVal})`;
				      this.context.fillRect(x, y, barWidth, barLength);
				    }
				  } else {
				    const formattedData = this.formatData(dataset); 
				    const xPadding =this.paddingX;
				    const yPadding = this.paddingY;
				    const xMax = this.canvas.width - xPadding;
				    const yMax = this.canvas.height - yPadding;

				    const maxVal = Math.max(...dataset.points.map(point => point.coordinates.y));

				    const barCount = formattedData.x.length;
				    const barWidth = (xMax - xPadding) / barCount;

				    for (let i = 0; i < barCount; i++) {
				      const x = this.origin.x + xPadding + i * barWidth;
				      const y = this.origin.y - formattedData.y[i];
				      const barHeight = formattedData.y[i];

				      const point = dataset.points[i];
				      let rVal, gVal, bVal, aVal;
				      if (point.coordinates.R != null && point.coordinates.G != null && point.coordinates.B != null) {
				        rVal = point.coordinates.R;
				        gVal = point.coordinates.G;
				        bVal = point.coordinates.B;
				        aVal = point.coordinates.A != null ? point.coordinates.A : 1.0; // default to fully opaque if alpha is not specified
				      } else {
				        rVal = 0;
				        gVal = 0;
				        bVal = 0;
				        aVal = 1.0;
				      }
				      this.context.fillStyle = `rgba(${rVal},${gVal},${bVal},${aVal})`;
				      this.context.fillRect(x, y, barWidth, barHeight);
				    }
				  }
				}
		 async drawScatterPlot(points, options, progress) {
			  const coordType = options.coordType;
			  let drawId = Math.random();
			  progress(drawId, 0, "Drawing Graph");
			  ////////////////console.log("DRAWING SCATTER PLOT", coordType);
			  const formattedData = coordType === 'Polar' ? this.formatPolarData(points) : this.formatData(points, options);
			  const xValues = coordType === 'Polar' ? formattedData.r : formattedData.x;
			  const yValues = coordType === 'Polar' ? formattedData.theta : formattedData.y;
			  const pointSize = options.width;

			  // draw points
const drawPoints = async (i) => {
  if (i < xValues.length) {
    progress(drawId, parseFloat(i) / xValues.length, "Drawing Graph");
    const point = points[i];
    let x, y, r, g, b, a;
    if (coordType === 'Polar') {
      x = this.origin.x + (xValues[i] * Math.cos(yValues[i]));
      y = this.origin.y - (xValues[i] * Math.sin(yValues[i]));
    } else {
      x = this.origin.x + (xValues[i]);
      y = this.origin.y - (yValues[i]);
    }
			    if (point.coordinates.R != null && point.coordinates.G != null && point.coordinates.B != null) {
			    	  r = point.coordinates.R;
			    	  g = point.coordinates.G;
			    	  b = point.coordinates.B;
			    	  a = point.coordinates.A != null ? point.coordinates.A : 1.0; // default to fully opaque if alpha is not specified
			    	} else {
			    	  r = DataSet.defaultColor('R');
			    	  g = DataSet.defaultColor('G');
			    	  b = DataSet.defaultColor('B');
			    	  a = DataSet.defaultColor('A');
			    	}
			    if(!x){
			    	x=0;
			    }
			    if(!y){
			    	y=0;
			    }
			    	this.context.beginPath();
			    	this.context.fillStyle = `rgba(${r},${g},${b},${a})`;
			    	this.context.arc(x, y, pointSize, 0, 2 * Math.PI);
			    	this.context.fill();
			    	this.hasChanged = true;
			    	
	//		    	setTimeout(() => {
     // drawPoints(i + 1, drawId, xValues, yValues, points, coordType, pointSize);
    //}, 0);
    //console.warn("drawing")
return 1;
  }
};
const worker = new MyWorker(null, drawPoints, "Drawing Points", xValues.length, 0);
worker.run();
// Start the loop
//drawPoints(0, drawId, xValues, yValues, points, coordType, pointSize);
			  
			}
			calculateColor(value) {
  const colorSpectrum = [
    "#9400D3", // violet
    "#4B0082", // indigo
    "#0000FF", // blue
    "#00FF00", // green
    "#FFFF00", // yellow
    "#FFA500", // orange
    "#FF0000"  // red
  ];

  const index = Math.round(value * (colorSpectrum.length - 1));
  return colorSpectrum[index];
}
drawText(text, x, y, color, font) {
  this.context.fillStyle = color;
  this.context.font = font;
  this.context.textAlign = 'center';
  this.context.textBaseline = 'middle';
  this.context.fillText(text, x, y);
}
		  drawIntervalChart(intervals, options) {
			  //const domainAxis = options.domainAxis;
			  //const rangeAxis = options.rangeAxes[0];

			  // Determine max and min values for domain and range axes
			  const maxX = Math.max(...intervals.map(interval => interval.end));
			  const minX = Math.min(...intervals.map(interval => interval.start));
			  const maxY = Math.max(...intervals.map(interval => interval.power));
			  const minY = 0;

			  // Calculate scaling factors
			  const xScaleFactor = ( this.canvas.width - 2 * this.paddingX)/(maxX - minX);
			  const yScaleFactor = (this.canvas.height - 2 * this.paddingY)/(maxY - minY);

			  // Set the origin of the graph
			  this.setOrigin(minX * xScaleFactor, minY * yScaleFactor);

			  // Loop through intervals and draw each one as a bar
			  intervals.forEach(interval => {
			    const x = interval.start * xScaleFactor;
			    const y = this.canvas.height - (interval.power * yScaleFactor);
			    const width = interval.duration * xScaleFactor;
			    const height = interval.power * yScaleFactor;

			    // Set color and pattern based on power value
			    const color = this.calculateColor(interval.power/maxY);
			    //const pattern = this.calculatePattern(interval.power);

			    // Draw the bar
			    this.context.fillStyle = color;
			    this.context.fillRect(x, y, width, height);
			    //////////////////////////console.warn(xScaleFactor, yScaleFactor, interval.duration, interval.power, x, y, width, height);

			    // Apply the pattern to the bar
			    //this.context.fillStyle = pattern;
			    //this.context.fillRect(x, y, width, height);

			    // Draw the value text
			    const textX = x + width / 2;
			    const textY = y + height / 2;
			    this.drawText(interval.power.toFixed(2), textX, textY, color, '12px sans-serif');
			  });
			}
			hslToRgb(h, s, l, channel) {
  const hueToRgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  let r, g, b;

  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hueToRgb(p, q, h + 1 / 3);
    g = hueToRgb(p, q, h);
    b = hueToRgb(p, q, h - 1 / 3);
  }

  const rgbValues = {
    R: Math.round(r * 255),
    G: Math.round(g * 255),
    B: Math.round(b * 255),
  };

  return rgbValues[channel];
}
progressOnGraph(id, value, label) {
  const progressBarId = `progress-bar-${id}`;
  let progressBar = document.getElementById(progressBarId);

  if (!progressBar && value < .995) {
    progressBar = document.createElement('div');
    progressBar.id = progressBarId;
    //progressBar.style.position = 'absolute';
    //progressBar.style.bottom = '0';
    //progressBar.style.left = '0';
    progressBar.style.height = '20px';
    progressBar.style.backgroundColor = 'rgba(0, 128, 255, 0.5)';
    this.onScreenCanvas.parentElement.appendChild(progressBar);
  }

  progressBar.style.width = `${value * this.canvas.width}px`;

  // If value is greater than 99.5, destroy the progress bar
  if (value > 0.995) {
    progressBar.remove();
  }
}
		  async drawGraph(data, options) {
			  await data;
			  const scaleMode = options.scaleMode;
			  const coordType = options.coordSystem;
			  const graphType = options.graphType;
			  let domainAxis = options.domainAxis;
			  let rangeAxes = options.rangeAxes;
			  let progress = options.progress;
			  if(!domainAxis){
				  domainAxis = 'x';
			  }
			  if(!rangeAxes){
				  rangeAxes = ['y'];
			  }
			  if(options.progress == null){
				  progress = this.progressOnGraph.bind(this);
			  }
this.dataTextArea.value = JSON.stringify(data, null, 2);
////////////////console.log(options);
			  // Determine if input data is an interval array or a point array
			  const intervalArray = data.intervals || (Array.isArray(data) && data.every(p => p instanceof Interval));
//////////console.error(intervalArray);
			  this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);

  const points = data.points || data;
  const totalRanges = rangeAxes.length;
  let nones = 0;
  for (let i = 0; i < totalRanges; i++) {
    if (rangeAxes[i] == "none") {
      nones++;
    }
	}
  for (let i = 0; i < totalRanges; i++) {
    if (rangeAxes[i] == "none") {
      continue;
    }

    let tempPoints = [];
    const hue = (i * 1) / (totalRanges-nones);
    for (let j = 0; j < points.length; j++) {
	if(totalRanges-nones > 1){

      tempPoints.push(
        new Point({
          x: points[j].coordinates[domainAxis],
          y: points[j].coordinates[rangeAxes[i]],
          R: this.hslToRgb(hue, 1, 0.5, "R"),
          G: this.hslToRgb(hue, 1, 0.5, "G"),
          B: this.hslToRgb(hue, 1, 0.5, "B"),
          A: points[j].coordinates["A"],
        })
      );
      }else{
		  tempPoints.push(new Point({x:points[j].coordinates[domainAxis], y:points[j].coordinates[rangeAxes[i]], R:points[j].coordinates['R'], G:points[j].coordinates['G'], B:points[j].coordinates['B'], A:points[j].coordinates['A']}));
		  }
	  }
    

			  if (intervalArray && graphType === 'interval') {
			    this.drawIntervalChart(intervalArray, coordType);
			  } else {
			    
			    switch (graphType) {
			      case 'scatter':
			        this.drawScatterPlot(tempPoints, options, progress);
			        break;
			      case 'line':
					  this.drawLineChart(tempPoints, options);
			        break;
			      case 'bar':
			        this.drawBarChart(tempPoints, coordType);
			        break;
			      default:
					  ////////////////////////console.warn(data);
			        throw new Error('Invalid graph type');;
			    }
			  }
			}
	}
}