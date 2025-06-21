function getHTMLControlBoxTemplate(id) {
	  let functionSelectorOptions = '';
	  
	  // loop through the options array to create select options
	  for (const option of availableFunctions) {
	    functionSelectorOptions += `<option value="${option.name}">${option.name}</option>`;
	  }
	  
	  return `<div>
	      <button id="add-before-${id}">Add Before</button>
	      <button id="add-after-${id}">Add After</button>
	      <button id="delete-${id}">Delete</button>
	    </div><div id="control-box-${id}" class="control-box">
		  
		      <div class="selector-container, input-controls">
		        <label for="input-type-${id}">Input</label>

		        
		      </div>
		      <div class="function-chain-editor-container">
		      <div class="function-chain-controls">
	          <label for="function-selector-${id}">Add Function</label>
	          <select id="function-selector-${id}">
	            ${functionSelectorOptions}
	          </select>
	          <button id="add-function-${id}">Add</button>
	        </div>  
		      <div class="function-chain-editor" id="function-chain-editor-${id}">
		          <!-- Placeholder for FunctionChain controls -->
		        
		      </div></div>
		      <div class="selector-container, output-controls">
		        <label for="output-type-${id}">Output</label>

		</div>`;
	}


class ControlBoxFactory {
	  constructor(id) {
	    this.container = document.createElement("div");
	    this.container.id = `controlbox-container-${id}`;
	    this.idPrefix = `interpolation-controlbox-${id}`;
	    const htmlTemplate = getHTMLControlBoxTemplate(id);
	    this.container.innerHTML = htmlTemplate;
////////////////////////////////////////console.log(JSON.stringify(this.container));
	    // Define functions for oninput, onchange, and onclick events
	    const onInputFunction = function () {
	      this.controlBox[this.id.substring(0, this.id.lastIndexOf("-"))] = this.value;
	      //verifyInputs();
	    };
	    const onChangeFunction = function () {
	      this.controlBox[this.id.substring(0, this.id.lastIndexOf("-"))] = this.value;
	      //verifyInputs();
	    };
	    const onClickFunction = function () {
	      this.controlBox[this.id.substring(0, this.id.lastIndexOf("-"))] = this.value;
	    };

	    const inputContainer = document.createElement("div");
	    inputContainer.id = `input-container-${id}`;
	    this.container.appendChild(inputContainer);
	    // Assign functions to input, select, and button elements
	    const elements = this.container.getElementsByTagName("*");
	    for (let i = 0; i < elements.length; i++) {
	      if (["INPUT", "SELECT", "BUTTON"].includes(elements[i].tagName)) {
	        elements[i].oninput = onInputFunction;
	        elements[i].onchange = onChangeFunction;
	        elements[i].onclick = onClickFunction;
	        elements[i].controlBox = this;
	      }
	    }
	  }
	}
	class ControlBox {
		  constructor(id, dataSet) {
			    this.id = id;
			    this.controlBoxManager = null;
			    ////////////////////////////////////////console.log(dataSet);
			    this.inputDataSet = dataSet;
			    this.outputDataSet = new DataSet();
			    let x = [];
			    let y = [];
			    if(this.inputDataSet && this.inputDataSet.points && this.inputDataSet.points.length > 2){
			    	////////////////////////////////////////console.log(JSON.stringify(this.inputDataSet));
     			    x = this.inputDataSet.getAllX();
	     		    y = this.inputDataSet.getAllY();
	     		    ////////////////////console.warn(y);
	     		   this.interpolator = new Interpolator(x,y);
			    }
			    
			    this.inputs = [];
			    this.outputs = [];
			    this.connectedInputs = [];
			    this.connectedOutputs = [];
			    this.div = new ControlBoxFactory(id).container;
				
			    //const canvas = document.createElement('canvas');
			    //canvas.width = 500;
			    //canvas.height = 500;
			    this.mainInputControl = new InputControl();
			    const mainControls = this.mainInputControl.createPreviewGraphOutput(1200,800);
			    
			  this.div.appendChild(mainControls);  
			    //this.dataTextArea = document.createElement('textarea');
			    //this.dataTextArea.style.width = '100%';
			    //this.dataTextArea.style.height = '200px';
			    //this.graph = new Graph(canvas, this.dataTextArea);
			    //this.div.appendChild(canvas);
			    //this.div.appendChild(this.dataTextArea);
			    this.functionChain = new FunctionChain(this);
//this.setInputDataSet  = this.setInputDataSet = this.setInputDataSet.bind(this);
	//			  this.getInput = this.getInput.bind(this);
			  }

		  setInputDataSet(dataSet) {
			  if(dataSet && DataSet.compare(this.inputDataSet, dataSet)){
	//			  ////////////////////////////////////console.warn("no change");
		//		  ////////////////////////////////////console.warn(dataSet);
			//	  ////////////////////////////////////console.warn(this.inputDataSet)
				  this.getOutputDataSet();
				  return;
				  
				  
			  }
		    this.inputDataSet = dataSet;
		    //////////////////////////////////////console.error(dataSet);
		    this.interpolator = new Interpolator(this.inputDataSet.getAllX(), this.inputDataSet.getAllY());
		    this.getOutputDataSet();
		  }

		  setOutputDataSet(dataSet) {
		    this.outputDataSet = dataSet;
		  }

		  setInterpolationMethod(method) {
		    this.interpolationMethod = method;
		  }
		  getSelectedFunction(){
			  const functionSelector = document.getElementById(`function-selector-${this.id}`);
			  const functionName = functionSelector.value;
		  
			  return functionName; 
		  }
		  async createFunctionControl(functionName) {
			  const functionContainer = document.createElement("div");
			  functionContainer.innerHTML = `<h3>${functionName}</h3>`;
			  const deleteButton = document.createElement("button");
			  const hideButton = document.createElement("button");
			  const upButton = document.createElement("button");
			  const downButton = document.createElement("button");
			  const buttonContainer = document.createElement("div");


			  functionContainer.className = "function-container";
			  upButton.textContent = "Up";
			  upButton.className = "up-button";
			  deleteButton.textContent = "Delete";
			  deleteButton.className = "delete-button";
			  downButton.textContent = "Down";
			  downButton.className = "down-button";
			  hideButton.textContent = "Hide";
			  hideButton.className = "hide-button";
			  

			  buttonContainer.appendChild(upButton);
			  buttonContainer.appendChild(deleteButton);
			  buttonContainer.appendChild(downButton);
			  buttonContainer.appendChild(hideButton);

			  
			  
			  
			  let instance;
			  const selectedFunction = availableFunctions.find(f => f.name === functionName);
			  ////////////////////////////////////console.log("LOOKING FOR "+selectedFunction.name);
			  let inputBoxes = [];
			  let inputs = [];
			  let inputControlReturn;
			  if (selectedFunction) {
			    instance = this.functionChain.getClassInstanceByName(selectedFunction.name);
			    const inputControl = new InputControl();

			    //from input control return statement:
			    //				  return {
//					    container: container,
//					    inputContainer: inputContainer,
//					    inputBoxes: this.inputBoxes,
//					    inputElements: inputElements,
//					    graphFunc: this.preview.bind(this),
//					  };
					 inputControlReturn = await inputControl.createInputs(instance, selectedFunction.forge.inputParams, selectedFunction.forge.inputTypes, selectedFunction.forge.labels);
					 inputs = inputControlReturn['inputElements'];
					 inputBoxes = inputControlReturn['inputBoxes'];
			  }
			    //instance.attachInputs(this.functionChain, functionParams, inputs);
			  /// for(let i = 0; i < inputBoxes.length; i++){
			  ///  functionContainer.appendChild(inputBoxes[i]);
			 /// }
			     functionContainer.appendChild(inputControlReturn['container']); 
			  

			  upButton.addEventListener("click", () => {
				  const functionChainEditor = document.getElementById(`function-chain-editor-${this.id}`);
				  const functionContainer = upButton.parentNode.parentNode;
				  const previousSibling = functionContainer.previousElementSibling;
				  functionChainEditor.insertBefore(functionContainer, previousSibling);
				  instance.moveUpChain(this.functionChain);
				  //////////////////////////////////////console.log("should have been moved up");
				});

				downButton.addEventListener("click", () => {

				  const functionChainEditor = document.getElementById(`function-chain-editor-${this.id}`);
				  const functionContainer = downButton.parentNode;
				  const nextSibling = functionContainer.nextElementSibling;
				  functionChainEditor.insertBefore(nextSibling, functionContainer);	
				  instance.moveDownChain(this.functionChain);
				});
			  deleteButton.addEventListener("click", () => {
				    this.functionChain.removeFunction(instance.id);
					  functionContainer.remove();
				  });
			  hideButton.addEventListener("click", () => {
				  let isHidden = (inputBoxes[0].style.display == 'none');
				    for(let i = 0; i < inputBoxes.length; i++){
				    	if(isHidden){
				    		inputBoxes[i].style.display = 'block';
				    	}else{
				    		inputBoxes[i].style.display = 'none';
				    	}
				    }
				    if(isHidden){
			    		hideButton.textContent = 'Hide';
			    	}else{
			    		hideButton.textContent = 'Show';
			    	}
				  });

				functionContainer.appendChild(buttonContainer);
			  
			  return [ functionContainer, instance, inputs, inputControlReturn.graphFunc ];
			}
		  async addFunctionControl() {
			  
			  const functionName = this.getSelectedFunction();
			  const [ functionControl, instance, inputs, graphFunc ] = await this.createFunctionControl(functionName);
			  const functionChainEditor = document.getElementById(`function-chain-editor-${this.id}`);
			  functionChainEditor.appendChild(functionControl);
			  instance.addToFunctionChain(this.functionChain);
			  instance.attachInputs(this.functionChain, instance.inputParams, inputs, graphFunc);
			  
			  //verifyInputs();

			}
		  
		  sendOutputDataSet(dataset){
			 // ////////////////////////////////////console.error("FOUND A NEW DATASET");
			  this.outputDataSet = dataset;
			  this.drawGraph(this.outputDataSet);//document.getElementById(`graph-type-${this.id}`).value, document.getElementById(`graph-coords-${this.id}`).value);
		  }
		  getInput(){
			  //////////////////////////////////////console.log("CONTROLBOX INPUT REQUESTED: "+JSON.stringify(this.inputDataSet));
			return this.inputDataSet;  
		  }
		  getOutputDataSet() {
			  
this.functionChain.setInput(this.inputDataSet);
			  ////////////////////////////////////////console.log(inputSparseSlider);
	//		  let outputDataSet = new DataSet();
			  //const intervals = this.inputDataSet.intervals;
//////////////////////////////////////console.log(this.inputDataSet);
			 // outputDataSet = this.functionChain.run(this.inputDataSet);  
//////////////////////////////////////console.log(outputDataSet);			  

  			  this.drawGraph(this.outputDataSet);//document.getElementById(`graph-type-${this.id}`).value, document.getElementById(`graph-coords-${this.id}`).value);
			//this.outputDataSet = outputDataSet;//(MathematicalOperation.lowPassFilter(outputDataSet, intervalSlider.value));
			return this.outputDataSet;
			
		  }

		  
		  drawGraph(graphType, coordType) {
			    this.mainInputControl.preview(this.outputDataSet);
			    //this.graph.drawGraph(this.outputDataSet, {graphType, coordSystem:coordType});
			  }
		  addInput(input) {
		    this.inputs.push(input);
		  }

		  addOutput(output) {
		    this.outputs.push(output);
		  }
		  connectInput(id) {
			  const inputBox = this.controlBoxManager.getBoxByID(id);
			  if (inputBox) {
			    this.connectedInputs.push(inputBox);
			    inputBox.connectedOutputs.push(this);
			  }
			}
		  connectOutput(id) {
			  const outputBox = this.controlBoxManager.getBoxByID(id);
			  if (outputBox) {
			    this.connectedOutputs.push(outputBox);
			    outputBox.connectedInputs.push(this);
			  }
			}
		  
		  disconnectInput(id) {
			  const inputBox = this.controlBoxManager.getBoxByID(id);
			  if (inputBox) {
			    const index = this.connectedInputs.indexOf(inputBox);
			    if (index !== -1) {
			      this.connectedInputs.splice(index, 1);
			      const outputIndex = inputBox.connectedOutputs.indexOf(this);
			      if (outputIndex !== -1) {
			        inputBox.connectedOutputs.splice(outputIndex, 1);
			      }
			    }
			  }
			}

		  disconnectOutput(id) {
			  const outputBox = this.controlBoxManager.getBoxByID(id);
			  if (outputBox) {
			    const index = this.connectedOutputs.indexOf(outputBox);
			    if (index !== -1) {
			      this.connectedOutputs.splice(index, 1);
			      const inputIndex = outputBox.connectedInputs.indexOf(this);
			      if (inputIndex !== -1) {
			        outputBox.connectedInputs.splice(inputIndex, 1);
			      }
			    }
			  }
		  }

		  sendData(data) {
		        if (this.connectedOutputs.length > 0) {
		            for (let i = 0; i < this.connectedOutputs.length; i++) {
		              this.connectedOutputs[i].receiveData(data);
		            }
		          } else {
		            //////////////////////////////////////console.log(`ControlBox ${this.id} has no connected outputs.`);
		          }
		        }

		        receiveData(data) {
		        	//////////////////////////////////////console.log(data);
		          this.setInputDataSet(data);
		          this.setOutputDataSet(this.getOutputDataSet());
		          this.sendData(this.outputDataSet);
		        }

		        update() {
		          this.setOutputDataSet(this.getOutputDataSet());
		          this.sendData(this.outputDataSet);
		        }
		      }
	class Connection {
		  constructor(controlBoxManager) {
		    this.controlBoxManager = controlBoxManager;
		  }

		  getCurrentOutputDataset() {
		    const controlBoxes = this.controlBoxManager.controlBoxes;
		    if (controlBoxes.length > 0) {
		      return controlBoxes[controlBoxes.length - 1].outputDataSet;
		    }
		    //////////////////////////////////////console.log('No control boxes connected');
		    return new DataSet();
		  }
		}

		class ControlBoxManager {
			  constructor(container, mainInput) {
			    this.controlBoxes = [];
			    this.connections = [];
			    this.container = container;
			    this.onControlBoxAdd = () => {};
			    this.onControlBoxRemove = () => {};
			  //  //////////////////////////////////////console.log(mainInput);

			    this.lastControlBox = this.addControlBox(mainInput, "after");
			  }
			  
			  setInputDataSet(dataset){
				  //////////////////////////////////////console.warn(dataset);
				  this.controlBoxes[0].setInputDataSet(dataset);
			  }

			  addControlBox(dataSet) {
				  const id = this.controlBoxes.length;
				  const controlBox = new ControlBox(id, dataSet);
			    controlBox.controlBoxManager = this;
			    this.controlBoxes.push(controlBox);
			    this.container.appendChild(controlBox.div);
			    this.onControlBoxAdd(controlBox);
			    
			    const addFunctionButton = document.getElementById(`add-function-${id}`);
			    addFunctionButton.addEventListener('click', () => {
			      controlBox.addFunctionControl(controlBox.getSelectedFunction());
			    });
			    // Connect the control box to the last added control box, or to the mainInput if this is the first control box.
			    if (this.controlBoxes.length > 1) {
			      const lastControlBox = this.controlBoxes[this.controlBoxes.length - 2];
			      lastControlBox.connectOutput(controlBox.id);
			    }

			    this.lastControlBox = controlBox;
			    return controlBox;
			  }

				  removeControlBox(controlBox) {
				    const index = this.controlBoxes.indexOf(controlBox);
				    if (index !== -1) {
				      this.controlBoxes.splice(index, 1);
				      this.container.removeChild(controlBox.div);
				      this.onControlBoxRemove(controlBox);
				      // Remove connections involving the removed control box.
				      for (let i = this.connections.length - 1; i >= 0; i--) {
				        const connection = this.connections[i];
				        if (connection.inputBox === controlBox || connection.outputBox === controlBox) {
				          this.connections.splice(i, 1);
				        }
				      }
				    }
				  }


				getControlBoxes(){
					return this.controlBoxes;
				}	  

			  

			  

			  getBoxInputs(id) {
			    const box = boxes.find((box) => box.id === id)?.element;
			    if (!box) {
			      return null; // or throw an error, depending on how you want to handle this case
			    }
			    const inputs = Array.from(box.querySelectorAll('input, select'));
			    const inputMap = {};
			    inputs.forEach((input) => {
			      const idParts = input.id.split('-');
			      const inputId = idParts.slice(0, -1).join('-');
			      const value = input.type === 'checkbox' ? input.checked : input.value;
			      inputMap[inputId] = value;
			    });
			    return inputMap;
			  }

			  getAllBoxInputs() {
			    const inputMaps = [];
			    boxes.forEach((box) => {
			      const inputs = Array.from(box.element.querySelectorAll('input, select'));
			      const inputMap = {};
			      inputs.forEach((input) => {
			        const idParts = input.id.split('-');
			        const inputId = idParts.slice(0, -1).join('-');
			        const value = input.type === 'checkbox' ? input.checked : input.value;
			        inputMap[inputId] = value;
			      });
			      inputMaps.push(inputMap);
			    });
			    return inputMaps;
			  }
			  updateConnections() {
			    this.connections = [];
			    for (let i = 0; i < this.controlBoxes.length - 1; i++) {
			      const fromBox = this.controlBoxes[i];
			      const toBox = this.controlBoxes[i + 1];
			      const connection = new Connection(fromBox.outputDataSet, toBox.inputDataSet);
			      this.connections.push(connection);
			    }
			  }

			  getControlBoxById(id) {
			    return this.controlBoxes.find(box => box.id === id);
			  }

			  createGraphs() {
			    this.container.innerHTML = '';
			    for (let i = 0; i < this.controlBoxes.length; i++) {
			      const controlBox = this.controlBoxes[i];
			      const graphContainer = document.createElement('div');
			      graphContainer.id = `graph-${controlBox.id}`;
			      graphContainer.classList.add('graph-container');
			      this.container.appendChild(graphContainer);
			      controlBox.createGraph(graphContainer, 'duration');
			    }
			  }
			}
		
		class ControlSystem {
			  constructor(dataSet = null) {
				  
				  ////////////////////////////////////////console.log(dataSet);
			    this.connections = [];
			    if(dataSet == null){
			    this.inputDataSet = new DataSet();
			  }else{
				  this.inputDataSet = dataSet;
			  }
			    this.outputDataSet = new DataSet();
				 this.container = document.getElementById('wave-modifier-container');
	//			 //////////////////////////////////////console.log(JSON.stringify(this.container));
		////////////////////console.log(this.inputDataSet);
				 this.controlBoxManager = new ControlBoxManager(this.container, this.inputDataSet);
			    this.connections[0] = new Connection(this.controlBoxManager);
				 const numControlBoxes = 2;
				  ////////////////////////////////////////console.log(dataSet);
			  }

			  addControlBox(controlBox) {
				  //////////////////////////////////////console.log("adding controlbox from system");
			    this.controlBoxManager.addControlBox(controlBox, after);
			  }

			  addConnection(connection) {
			    this.connections.push(connection);
			  }

			  setInputDataSet(dataSet) {
				//  ////////////////////////////////////console.warn(dataSet);
			    this.inputDataSet = dataSet;
			    this.controlBoxManager.setInputDataSet(dataSet);
			  }

			  setOutputDataSet(dataSet) {
			     this.outputDataSet = dataSet;
			   // for (const box of this.controlBoxes) {
			   //   box.setOutputDataSet(dataSet);
			  //  }
			  }
			  getCurrentOutputDataset() {
				  //////////////////////////////////////console.log(this.connections[0].getCurrentOutputDataset());
				    return this.connections[0].getCurrentOutputDataset();
				  }
				getControlBoxes(){
					return this.controlBoxManager.getControlBoxes();
				}	

		}