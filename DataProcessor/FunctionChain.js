class FunctionChainLink {
	  constructor() {
	    this.fn = null;
	    this.paramNames = [];
	    this.inputTypes = [];
	    this.labels = [];
	    this.inputs = [];
	    this.updatedInput = false;
	    this.updatedOutput = false;
	    this.inputData;
	    this.outputData;
	    this.userInput;
	    this.persistentData = {};
	    this.inputControl;
	    this.progress;
	    this.processID;
	    this.fc;
	//    this.updateInputs = this.updateInputs.bind(this);
	    this.verifyInputs = this.verifyInputs.bind(this);
	  this.setPreview = this.setPreview.bind(this);
	  this.unSetPreview = this.unSetPreview.bind(this);
	  }
	  setPreview(){
		  this.fc.runResolution = 50;
	  }
	  
	  async unSetPreview(){
		  //this.inputData = null;
		  this.fc.runResolution  = 1;
		 // await this.updateInputs(this.fc.getFunctionInput(this.id), this.fc, true);
	  }

	  async updateInputs(dataset, fc = null, force = false){
		 ////console.log("LINK RECEIVING UPDATE");
		  if(fc){
			  this.fc = fc;
		  }
		  
		  if(!force && this.inputData && dataset.points && dataset.points[0] && dataset.points[0].coordinates && DataSet.compare(this.inputData, dataset)){
			////console.log("NO DICE, NO CHANGE", this.inputData, dataset); 
		  	this.updatedInput = false;
		  }else{
			  
			  this.inputData = dataset;
			  this.updatedInput = true;
		  }		  
		  if(fc){
		  let userInput = fc.getUserInput(this.id);
		 ////console.error(userInput);
		  if(!this.updatedInput && JSON.stringify(userInput) != JSON.stringify(this.userInput)){
			////console.warn("WAIT NO THE USER INPUT CHANGED");  
			  this.updatedInput = true;
			  this.userInput = userInput;
		  }
		  
		  ///TODO: implement a thread and mutex to handle this, the time while output data is being generated.
		  
		  if(this.updatedInput){
			  
			  ////console.log("GETTING MY OUTPUT");
		  let output = await this.fc.runIndividual(this.id);
		  if(!this.outputData || !DataSet.compare(output, this.outputData)){
			  this.outputData = output; 
			  this.updatedOutput = true;
		  }else{
			  this.updatedOutput = false;
		  }
		  this.updatedInput = false;
		  }
		  
		  if(this.updatedOutput){
			 ////console.log("LINK IS PUSHING OUTPUT") 
			  ////console.warn(this.outputData);
			  this.fc.notifyOutputChange(this.outputData, this.id);
			  this.updatedOutput = false;
		  }
		  }
	  }
	  async verifyInputs(){
		  ////console.log("EVENT NOTICED");
		  await this.updateInputs(this.inputData, this.fc);

	  }	  
	  setFunction(fn, paramNames, inputTypes, labels) {
	    this.fn = fn;
	    this.paramNames = paramNames || [];
	    this.inputTypes = inputTypes || [];
	    this.labels = labels || [];
	  }

	  addToFunctionChain(fc) {
		this.fc = fc;
	    this.id = fc.addFunction(this, this.fn, this.paramNames, this.inputTypes, this.labels);
	  }
	  moveUpChain(fc){
		  fc.moveFunctionUp(this.id);
	  }
	  moveDownChain(fc){
		  fc.moveFunctionDown(this.id);
	  }
	  
	  attachInputs(fc, paramNames, inputs, previewFunc){
			this.fc = fc;
		  let param = { paramNames, inputs };
		  this.previewFunc = previewFunc;
		  
		  this.inputs.push( param );
		  ////////////////////////////////////console.warn(this.inputs);
		  fc.addInputsArray(this.id, this.inputs[0].inputs);
	  }
	}
	class FunctionChainLinkForge {
	static createLinkClass(linkData) {
		  const LinkClass = class extends FunctionChainLink {
			  static typeName = linkData.name || null;
			  constructor() {
		      super(linkData.inputControl);
		      this.paramNames = linkData.paramNames || [];
		      this.inputTypes = linkData.inputTypes || [];
		      this.labels = linkData.labels || [];
		      this.inputParams = linkData.inputParams || [];
		      
		      //////////////////////////////////////console.error(this.name);

		      this.setFunction(linkData.fn, this.paramNames, this.inputTypes, this.labels);

		      if (linkData.customMethods) {
		        for (const methodName of linkData.customMethods) {
		          const methodDef = functionDefinitions[methodName];
		          if (methodDef) {
		            this[methodName] = methodDef.fn.bind(this);
		          } else {
		            ////////////////////////////////////console.warn(`Function definition for "${methodName}" not found`);
		          }
		        }
		      }
		    }
		  };

		  LinkClass.linkData = linkData;
		  return LinkClass;
		}

	  static createLinkClasses(linkDataArray) {
	    const linkClasses = [];
	    for (const linkData of linkDataArray) {
	      const LinkClass = this.createLinkClass(linkData);
	      linkClasses.push(LinkClass);
	    }
	    return linkClasses;
	  }
	}


class FunctionChain {
	  constructor(controlbox) {
		  this.controlbox = controlbox;
		  this.inputData = null;
		  this.outputData = null;
		  this.instanceCache = [];
	    this.functions = [];
	    this.previewFunctions = [];
	    this.inputs = [];
	    this.ids = [];
	    this.paramValues = [];
	    this.runResolution = 1;
	    this.linkClasses = FunctionChain.createLinks(availableFunctions, functionDefinitions);
	    ////////////////////////////////////console.warn(this.linkClasses);
	    //this.getFunctionInput = this.getFunctionInput.bind(this);
	 
	  }
	  static createLinks(availableFunctions, functionDefinitions) {
		  const linkClasses = [];
		  for (const availableFn of availableFunctions) {
		    const [selectedFnName, selectedFn] = Object.entries(functionDefinitions).find(([name, fn]) => name === availableFn.forge.fn) || [];
		    if (selectedFn) {
		      const linkData = { ...availableFn.forge, name: availableFn.name, fn: selectedFn };
		      const linkClass = FunctionChainLinkForge.createLinkClass(linkData);
		      linkClasses.push(linkClass);
		    }
		  }
		  return linkClasses;
		}
	  getClassInstanceByName(name) {
		 // ////////////////////////////////////console.log("Searching for class with name: ", name);
		  const linkClass = this.linkClasses.find(cls => {
		//	  ////////////////////////////////////console.log("Checking class with name: ", cls.typeName);
			  return cls.typeName === name;
		  });
		  
		  if (linkClass) {
		  //  ////////////////////////////////////console.log("Found class with name: ", linkClass.name);
		    const instance = new linkClass();
		    // this.instanceCache.push(instance);
		    return instance;
		  }
		  
		  //////////////////////////////////////console.error("Could not find class with name: ", name);
		  return null;
		}
	  addToCache(functionLink, id){
		  this.instanceCache.push(functionLink);
		  this.instanceCache[this.instanceCache.length-1].id = id;
		//  ////////////////////////////////////console.warn(this.instanceCache);
		//  ////////////////////////////////////console.warn(this.findCacheIndex(id));
	  }
	  
	  addFunction(functionLink, fn, paramNames, inputTypes, labels) {
		  const id = Symbol();
		  this.ids.push(id);
		  this.previewFunctions.push();
		  this.addToCache(functionLink, id);
	    this.functions.push(fn);
	    this.inputs.push({ paramNames, inputTypes, labels, inputs: []});
	    //////////////////////////////////////console.log(this.inputs[this.inputs.length-1]);
	    
		return id;

	  }

	  updateOutput(newOutput){
		  ////console.log("receiving chain output: "+JSON.stringify(newOutput));
		  if(!this.outputData || !DataSet.compare(newOutput, this.outputData)){
			  ////////////////////////////////////console.log("sending chain output.");
		  this.outputData = newOutput;
		  //////////////////////////console.error(newOutput);
		  this.controlbox.sendOutputDataSet(this.outputData);
		  }
	  }
	  async setInput(dataset){
		  
		  this.inputData = dataset;
		  if(this.ids[0]){
			 let cacheIndex = this.findCacheIndex(this.ids[0]);
			 await this.instanceCache[cacheIndex].updateInputs(this.inputData, this);
		  }else{
			  this.updateOutput(dataset);
		  }
}
	  getFunctionInput(id){
		  let index = this.ids.indexOf(id) - 1;
		  if(index == -1){
			  return this.inputData;
		  }
		  const cacheIndex = this.findCacheIndex(this.ids[index]);
		  ////////////////////////////console.warn("RETURNING PREVIOUS LINK DATA: ", this.runResolution);
		  return this.instanceCache[cacheIndex].outputData;
		  
		  
	  }
	  async getOutput(){
		  if(this.ids.length == 0){
				  this.inputData = this.controlbox.getInput(); 
				  return this.inputData
		  }
		  return await this.runIndividual(this.ids[this.ids.length - 1]);
	  }
	  async runIndividual(id) {
  const cacheIndex = this.findCacheIndex(id);
  if (cacheIndex == -1) {
    // ////////////console.error("instance not found in cache");
  }
  let inputData = this.instanceCache[cacheIndex].inputData || this.getFunctionInput(id);

  const oldOutputData = JSON.stringify(this.instanceCache[cacheIndex].outputData);

  const index = this.ids.indexOf(id);

  if (!inputData) {
    if (index == 0) {
      // ////////////console.error("LOADING FROM CONTROLLER");
      inputData = this.inputData;
      if (!inputData) {
        inputData = this.controlbox.getInput();
        if(!inputData){
			inputData = new DataSet();
		}
        this.instanceCache[cacheIndex].inputData = DataSet.clone(inputData);
      }
    } else {
      // ////////////console.log("RECURSIVELY FINDING INPUT");
          inputData = await this.runIndividual(this.ids[index - 1]);
	      return;
	    
    }
  }

  let returnVal = await this.run(inputData, index, 1);

  if (index == this.ids.length - 1) {
	  
    this.updateOutput(returnVal);
  } else if (!oldOutputData || JSON.stringify(returnVal) != oldOutputData) {
    // ////////////console.log("PROPAGATING UPDATED DATA");
    let nextId = this.ids[this.ids.indexOf(id) + 1];
    let nextCacheIndex = this.instanceCache.findIndex((entry) => entry.id == nextId);
    await this.instanceCache[nextCacheIndex].updateInputs(returnVal, this);
  } else {
    // ////////////console.log("run(inputData, index, 1) returned the same as instance.outputData");
  }

  return returnVal;
}
	  findCacheIndex(id){
		  return this.instanceCache.findIndex(entry => entry.id == id);
	  }
	  
	  getUserInput(id) {
    const inputInstance = this.inputs[this.ids.indexOf(id)];
    if (!inputInstance) {
      throw new Error(`No input instance found with id ${id}`);
    }

    const inputs = inputInstance.inputs;
    return FunctionChain.getUserInputStatic(inputs);
  }
	   static getUserInputStatic(inputs) {
  let params = [];
  let nameHistory = [];

  inputs.forEach((input) => {
	  
	  console.log(input);
    if (input.type === "checkbox") {
      if (inputs.filter((inp) => inp.name === input.name).length > 1) {
        if (!nameHistory.includes(input.name)) {
          params.push(inputs.filter((inp) => inp.name === input.name && inp.checked).map((inp) => inp.value));
          nameHistory.push(input.name);
        }
      } else {
        params.push(input.checked);
      }
    } else if (input.type === "radio") {
      if (inputs.filter((inp) => inp.name === input.name).length > 1) {
        if (!nameHistory.includes(input.name)) {
          params.push(document.querySelector(`input[name="${input.name}"]:checked`).value);
          nameHistory.push(input.name);
        }
      } else {
        params.push(input.value);
      }
    } else if (input.type === "file") {
      params.push(input.files[input.files.length - 1]);
    } else if (input?.tagName?.toLowerCase() === "select") {
      params.push(input?.options[input?.selectedIndex]?.value);
    } else {
      params.push(input.value);
    }
  });

  return params.filter((param) => param !== undefined);
}
	  async notifyOutputChange(newData, id){
		  //run(newData, this.ids.indexOf(id), -1);

		  
		  let nextId = 1 + this.ids.indexOf(id);
		  if(nextId >= this.ids.length){
			  this.updateOutput(newData);
		  }else{
			  nextId = this.ids[nextId];
		 	 let cacheIndex = this.instanceCache.findIndex(entry => entry.id == nextId);
		 	 if(cacheIndex == -1){
		 		 ////////////////////////////////////console.error("COULDN'T FIND INSTANCE IN CACHE:" + this.instanceCache);
		 	 }
		 	 ////////////////////////////////////console.log("NOTIFYING NEXT LINK");
		 	 await this.instanceCache[cacheIndex].updateInputs(newData, this);
		 }
			  
	  }
	  refresh(){
		  this.run(this.inputData);
	  }
	 run(inputData, offset = 0, run = -1) {
  if(run == -1){
    run = this.ids.length - offset;
    if(run < 0 || offset < 0 || offset > this.ids.length - 1){
      return inputData;
    }
  }
 //////////console.error(inputData);
 	
 	let output = null;
 	if(inputData){
  		output = DataSet.clone(inputData);//.prune(this.runResolution);
	}else{
		//console.warn("NO INPUT DATA");
		output = new DataSet();
	}

  const processFn = async (link, fn, output, inputs) => {
	
    const result = fn(link, output, ...inputs);
    return result;
  };

  const runAsync = async () => {
    for (let i = offset; i < this.functions.length && i < offset+run; i++) {
      let cacheIndex = this.findCacheIndex(this.ids[i]);

      const fn = this.functions[i];
      const inputs = this.getUserInput(this.ids[i]);
      ////console.warn(inputs);
//console.warn("running a process, ", cacheIndex);
      output = await processFn(this.instanceCache[cacheIndex], fn, output, inputs);
      this.instanceCache[cacheIndex].previewFunc(output);
    }
    return output;
  };

  return runAsync();
}
	  addInputsArray(id, inputs){
		  const index = this.ids.indexOf(id);
		  this.inputs[index].inputs = inputs;
		  //inputs[0].dispatchEvent(new Event('change'));
	  }
	  		  
	  
	  removeFunction(id) {
		  const index = this.ids.indexOf(id);
		    if(index !== -1){
		    	this.functions.splice(index, 1);
		    	this.inputs.splice(index, 1);
		    	this.ids.splice(index, 1);
		    }
		  }

		  insertFunction(functionLink, index, fn, paramNames, inputTypes, labels) {
		    const id = Symbol();
			  this.functions.splice(index, 0, fn);
		    this.inputs.splice(index, 0, { paramNames, inputTypes, labels, inputs: [] });
		    this.ids.splice(index, 0, id);
		    this.addToCache.push(functionLink);
		    return index; //this will need to be fixed to shift indexes
		  }
		  
		  moveFunctionUp(id) {
			    const index = this.ids.indexOf(id);
			    if (index > 0) {
			      // swap the function, inputs, and id with the previous one
			      [this.functions[index], this.functions[index-1]] = [this.functions[index-1], this.functions[index]];
			      [this.inputs[index], this.inputs[index-1]] = [this.inputs[index-1], this.inputs[index]];
			      [this.ids[index], this.ids[index-1]] = [this.ids[index-1], this.ids[index]];
			    }
			    this.refresh();
			  }

			  moveFunctionDown(id) {
			    const index = this.ids.indexOf(id);
			    if (index < this.ids.length-1) {
			      // swap the function, inputs, and id with the next one
			      [this.functions[index], this.functions[index+1]] = [this.functions[index+1], this.functions[index]];
			      [this.inputs[index], this.inputs[index+1]] = [this.inputs[index+1], this.inputs[index]];
			      [this.ids[index], this.ids[index+1]] = [this.ids[index+1], this.ids[index]];
			    }
			    this.refresh();
			  }
	}


