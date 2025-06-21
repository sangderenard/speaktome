class GCMConversation {
  constructor(hashedPassword) {
	this.sessionID = Math.floor(Math.random()*1000000);
	this.hashedPassword = hashedPassword;
    this.pbkPassword = window.crypto.getRandomValues(new Uint8Array(32));
    this.ivBlock = null;
    this.messageNumber = 0;
    this.gcmInitializationParams = null;
    //this.publicKey = null;
    this.tagLength = 256;
  }

  async initialize() {
	//this.publicKey = await ServerCommunication.getPublicKey(this.sessionID);
    this.ivBlock = window.crypto.getRandomValues(new Uint8Array(16 * 32));
await ServerCommunication.testEncryptDecrypt();

  }



async updateGcmInitializationParams() {
  await this.initialize();
  const pbkPassword = this.pbkPassword;
  const salt = window.crypto.getRandomValues(new Uint8Array(16));
  const iv = window.crypto.getRandomValues(new Uint8Array(16));
  //const configIV = await ServerCommunication.deriveGcmIvFromKey(await ServerCommunication.getSharedSecret(this.sessionID));


  const buffer = new ArrayBuffer(64);
const view = new DataView(buffer);

let offset = 0;

// Write the salt to the buffer
salt.forEach((value, index) => {
  view.setUint8(offset + index, value);
});
offset += salt.byteLength;

// Write the PBKDF2 password to the buffer
pbkPassword.forEach((value, index) => {
  view.setUint8(offset + index, value);
});
offset += pbkPassword.byteLength;

// Write the IV to the buffer
iv.forEach((value, index) => {
  view.setUint8(offset + index, value);
});
offset += iv.byteLength;
const byteLength = new TextEncoder().encode(buffer).length;
console.log(byteLength);
const { keyPair, publicKeyPem } = await ServerCommunication.generateRsaKeyPair();
  //const encryptedConfig = await ServerCommunication.encryptWithECC(await ServerCommunication.getSharedSecret(this.sessionID), configIV, dataConfigJson);
  const {serverPublicKey, serverSuggestions} = await ServerCommunication.getPublicKey(this.sessionID, publicKeyPem);
  const encryptedConfig = await ServerCommunication.encryptWithRSA(serverPublicKey, buffer);
  console.log("This is our encrypted binary data: ", encryptedConfig);
  const encryptedConfigB64 = btoa(new Uint8Array(encryptedConfig));
  console.log("This is our encrypted config data: ", encryptedConfigB64);
  const encryptedConfigUri = encodeURIComponent(encryptedConfigB64);

  const derivedKey = await ServerCommunication.gcmFromPBK(pbkPassword, salt);
  const encryptedData = await ServerCommunication.encryptWithGCM(derivedKey, new Uint8Array(this.ivBlock), iv);
  const encryptedDataB64 = btoa(new Uint8Array(encryptedData));
  console.log("This is our encrypted random data: ", encryptedDataB64);
  const encryptedDataUri = encodeURIComponent(encryptedDataB64);

  const url = `https://bike.dogbite.me/cgi-bin/confirmGCM.php?sessionID=${this.sessionID}&encryptedData=${encryptedDataUri}&encryptedConfig=${encryptedConfigUri}`;
  
 try {
  const response = await fetch(url);
  const encryptedCrcWithTagB64 = await response.text();
  console.warn(encryptedCrcWithTagB64);
  const encryptedCrcWithTag = Uint8Array.from(atob(encryptedCrcWithTagB64), c => c.charCodeAt(0));
  const encryptedCrc = encryptedCrcWithTag.subarray(0, encryptedCrcWithTag.length - this.tagLength / 8);
  const tag = encryptedCrcWithTag.subarray(encryptedCrcWithTag.length - this.tagLength / 8);
  const crc = await ServerCommunication.decryptWithGCM(derivedKey, encryptedCrc, iv, tag);
  if (crc.length === 32 && ServerCommunication.arrayBuffersEqual(crc, ServerCommunication.sha256(this.ivBlock))) {
    this.messageNumber = 0;
  } else {
    console.error('Invalid CRC');
  }
} catch (error) {
  console.error(error);
}
}




  async sendMessage(message) {
    const salt = this.saltBlock.slice(this.messageNumber * 16, (this.messageNumber + 1) * 16);
    const derivedKey = await ServerCommunication.deriveGcmKey(this.gcmInitializationParams.iv, this.hashedPassword, salt);
    const encryptedMessage = await ServerCommunication.encryptWithGCM(derivedKey, message, salt);
    // Send encrypted message to server
    this.messageNumber++;
    if (this.messageNumber === 16) {
      await this.updateGcmInitializationParams();
    }
  }

  async receiveMessage(encryptedMessage) {
    const salt = this.saltBlock.slice(this.messageNumber * 16, (this.messageNumber + 1) * 16);
    const derivedKey = await ServerCommunication.deriveGcmKey(this.gcmInitializationParams.iv, this.hashedPassword, salt);
    const message = await ServerCommunication.decryptWithGCM(derivedKey, this.gcmInitializationParams, encryptedMessage, salt);
    this.messageNumber++;
    if (this.messageNumber === 16) {
      await this.updateGcmInitializationParams();
    }
    return message;
  }
}

class ServerCommunication{
	constructor(password){
	
		this.hashedPassword = CryptoJS.SHA256(password).toString();
		
		this.conversation = new GCMConversation(this.hashedPassword);
		this.testServerCommunication();
	}
	async testServerCommunication(){
		this.conversation.updateGcmInitializationParams();
	}
	static async deriveKeyAndIV(input, salt, hash = 'SHA-256') {
  const keySize = 32; // 32 bytes = 256 bits
  const ivSize = 16; // 16 bytes = 128 bits
  const iterations = 1; // number of iterations

  const encodedInput = new TextEncoder().encode(input);
  const encodedSalt = new TextEncoder().encode(salt);

  const keyMaterial = await window.crypto.subtle.importKey(
    'raw',
    encodedInput,
    { name: 'PBKDF2' },
    false,
    ['deriveBits']
  );

  const params = { name: 'PBKDF2', salt: encodedSalt, iterations: iterations, hash: hash };
  const derivedBits = await window.crypto.subtle.deriveBits(params, keyMaterial, (keySize + ivSize) * 8);
  const derivedBytes = new Uint8Array(derivedBits);

 // const key = derivedBytes.slice(0, keySize);
//  const iv = derivedBytes.slice(keySize, keySize + ivSize);
const key = new Uint8Array(atob('7xv2Cq3V8H50zQhOED7wv/gO/umG8dL42iCHrN52NvI=').split('').map(char => char.charCodeAt(0)));
const iv = new Uint8Array(atob('IaU6z4UwPsdU6DZUdHYH2Q==').split('').map(char => char.charCodeAt(0)));
//console.warn("The key:iv input: ", '7xv2Cq3V8H50zQhOED7wv/gO/umG8dL42iCHrN52NvI=', ":", 'IaU6z4UwPsdU6DZUdHYH2Q==');
console.warn("The key:iv output: ", btoa(String.fromCharCode.apply(null, key)), ":", btoa(String.fromCharCode.apply(null, iv)));
 
  return { key, iv };
}

static async simpleEncrypt(data, salt, passcode, cipher = 'aes-256-gcm', hash = 'SHA-256') {
	try{
  const derived = await ServerCommunication.deriveKeyAndIV(passcode, salt, hash);
  const encodedData = new TextEncoder().encode(data);
  const encodedSalt = new TextEncoder().encode(salt);
  const iv = derived.iv;

  let algorithm;

  switch (cipher) {
    case 'aes-256-cbc':
      algorithm = { name: 'AES-CBC', length: 256, iv: iv };

      break;
    case 'aes-256-ctr':
      algorithm = { name: 'AES-CTR', length: 256, counter: iv, length: 128 };

      break;
    case 'aes-256-gcm':
      algorithm = { name: 'AES-GCM', length: 256, iv: iv, tagLength: 128};

      break;
    default:
      throw new Error('Invalid cipher');
  }
    const key = await window.crypto.subtle.importKey(
      'raw',
      derived.key,
      algorithm,
      false,
      ['encrypt']
    );
  let encryptedData;

  if (cipher === 'aes-256-gcm') {
  const tagLength = 128;
const encrypted = await window.crypto.subtle.encrypt(algorithm, key, encodedData);
const encryptedBytes = new Uint8Array(encrypted);
const tag = encryptedBytes.slice(-tagLength / 8);
encryptedData = new Uint8Array(encryptedBytes.slice(0, -tagLength / 8));
const returnVal = `${btoa(String.fromCharCode(...encryptedData))}:${btoa(String.fromCharCode(...tag))}`;

return returnVal;
  } else {
	  const tag = window.crypto.getRandomValues(new Uint8Array(16));
    const encrypted = await window.crypto.subtle.encrypt(algorithm, key, encodedData);
    encryptedData = new Uint8Array(encrypted);
    return `${btoa(String.fromCharCode(...encryptedData))}:${btoa(String.fromCharCode(...tag))}`;
  }
    }catch(error){
	  console.error("encrypt error: ", error);
  }
}

static async simpleDecrypt(data, salt, passcode, cipher = 'aes-256-gcm', hash = 'SHA-256') {
	try{
  const derived = await ServerCommunication.deriveKeyAndIV(passcode, salt, hash);
    const encodedSalt = new TextEncoder().encode(salt);
  const iv = derived.iv;
  
const dataParts = data.split(':');
console.log("key and tag count: ", dataParts.length)
let encryptedData;

if (cipher === 'aes-256-gcm') {
  const combinedData = atob(dataParts[0]) + atob(dataParts[1]);
  encryptedData = Uint8Array.from(combinedData, c => c.charCodeAt(0));
} else {
  encryptedData = Uint8Array.from(atob(dataParts[0]), c => c.charCodeAt(0));
}

const tag = Uint8Array.from(atob(dataParts[1]), c => c.charCodeAt(0));
console.warn("The encrypted data and tag: ",btoa(String.fromCharCode.apply(null, encryptedData)) , ":", btoa(String.fromCharCode.apply(null, tag)));

  let tagLength = 128;
let algorithm;
  switch (cipher) {
    case 'aes-256-cbc':
      algorithm = { name: 'AES-CBC', length: 256, iv: iv };

      break;
    case 'aes-256-ctr':
      algorithm = { name: 'AES-CTR', length: 256, counter: iv, length: 128 };

      break;
    case 'aes-256-gcm':
		 algorithm = { name: 'AES-GCM', length: 256, iv: iv, tagLength };
      break;
    default:
      throw new Error('Invalid cipher');
  }
  let key;
  try{
    key = await window.crypto.subtle.importKey(
      'raw',
      derived.key,
      algorithm,
      false,
      ['decrypt']
    );
    }catch(error){
		console.error("Error importing key: ", error);	
	}
  let decryptedData;

try{
  if (cipher === 'aes-256-gcm') {
   
    

decryptedData = await window.crypto.subtle.decrypt(
    algorithm,
    key,
    encryptedData,
    tagLength,
    tag
);

  } else {
   
    decryptedData = await window.crypto.subtle.decrypt(
      algorithm,
      key,
      encryptedData,
      
    );
  }
}catch(error){
	console.error("Decryption failure: ", error);
}
  const decodedData = new TextDecoder().decode(decryptedData);
  return decodedData;
  }catch(error){
	  console.error("decrypt error: ", error);
  }
}

static async testEncryptDecrypt() {
  const hashAlgorithms = ['SHA-256', 'SHA-384', 'SHA-512'];
  const ciphers = [
    'aes-256-ctr',
    'aes-256-gcm',
    'aes-256-cbc',
  ];

  const passcode = window.crypto.getRandomValues(new Uint8Array(16));
  const data = new Uint8Array(756);
for (let i = 0; i < 756; i++) {
  data[i] = i % 256;
}
  const salt = window.crypto.getRandomValues(new Uint8Array(16));

  for (const hashAlgorithm of hashAlgorithms) {
    for (const cipher of ciphers) {
      const encrypted = await ServerCommunication.simpleEncrypt(
        data,
        salt,
        passcode,
        cipher,
        hashAlgorithm
      );
      const decrypted = await ServerCommunication.simpleDecrypt(
        encrypted,
        salt,
        passcode,
        cipher,
        hashAlgorithm
      );
      if (btoa(decrypted) === btoa(data)) {
        console.log(`PASS: ${cipher} with ${hashAlgorithm}`);
      } else {
        console.log(`FAIL: ${cipher} with ${hashAlgorithm}`);
      }
    }
  }
}

static async gcmFromPBK(password, salt ) {
  const iterations = 100000;
  const passwordBytes = new TextEncoder().encode(password);
  const baseKey = await window.crypto.subtle.importKey(
    "raw",
    passwordBytes,
    { name: "PBKDF2" },
    false,
    ["deriveBits"]
  );

  const derivedBits = await window.crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      salt: salt,
      iterations: iterations,
      hash: { name: "SHA-256" },
    },
    baseKey,
    256 // 256-bit key length
  );

  const key = await window.crypto.subtle.importKey(
    "raw",
    derivedBits,
    { name: "AES-GCM" },
    false,
    ["encrypt", "decrypt"]
  );

  return key;
}

static async encryptWithGCM(key, data, iv) {	
  const algorithm = { name: 'AES-GCM', iv: iv, tagLength: 128 };
  return await crypto.subtle.encrypt(algorithm, key, data);
}
static async encryptWithRSA(publicKey, data) {
	try{
  return await window.crypto.subtle.encrypt({
    name: 'RSA-OAEP'
  }, publicKey, new TextEncoder().encode(data));
  }catch(error){
	  console.error(error);
  }
}
static async encryptWithECC(gcmKey, iv, data) {
  
  const encodedData = new TextEncoder().encode(data);
  const encryptedData = await ServerCommunication.encryptWithGCM(gcmKey, encodedData, iv);
  return encryptedData;
}
static async deriveGcmIvFromKey(key) {
  const keyData = await crypto.subtle.exportKey('raw', key);
  const hashedData = CryptoJS.SHA256(CryptoJS.lib.WordArray.create(keyData));
  const salt = CryptoJS.lib.WordArray.create(hashedData.words.slice(-4));
  const iterations = 100000;
  const keyLength = 16;
  const derivedKey = CryptoJS.PBKDF2(keyData, salt, { keySize: keyLength / 4, iterations: iterations });
  const iv = new Uint8Array(derivedKey.words.slice(-4));
  //const iv = crypto.getRandomValues(new Uint8Array(12));
  return iv;

}

static async getSharedSecret(sessionID) {
  const { publicKey, privateKey, publicKeyPEM } = await ServerCommunication.generateEcKeyPair();
  const importedPeerPublicKey = await ServerCommunication.getPublicKey(sessionID, publicKeyPEM);
  const importedPrivateKey = await ServerCommunication.importJWK(privateKey, ['deriveKey']);
 // const importedPeerPublicKey = await ServerCommunication.importJWK(peerPublicKey, ['deriveKey']);

  const sharedSecret = await crypto.subtle.deriveKey(
    { name: 'ECDH', public: importedPeerPublicKey },
    importedPrivateKey,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  );

  return sharedSecret;
}
static async generateEcKeyPair() {
  const curve = 'P-521';
  const algorithm = { name: 'ECDH', namedCurve: curve };
  const extractable = true;
  const keyUsages = ['deriveKey', 'deriveBits'];
  const keyPair = await window.crypto.subtle.generateKey(algorithm, extractable, keyUsages);
  const publicKey = await window.crypto.subtle.exportKey('jwk', keyPair.publicKey);
  const privateKey = await window.crypto.subtle.exportKey('jwk', keyPair.privateKey);
  const publicKeyPEM = `-----BEGIN PUBLIC KEY-----\n${ServerCommunication.base64ToPem(publicKey.x + publicKey.y, 64)}\n-----END PUBLIC KEY-----\n`;
  return { publicKey, privateKey, publicKeyPEM, keyPair };
}
static async generateRsaKeyPair() {
  const algorithm = { name: 'RSA-OAEP', hash: 'SHA-512', modulusLength: 4096,  publicExponent: new Uint8Array([0x01, 0x00, 0x01]) };
  const extractable = true;
  const keyUsages = ['encrypt', 'decrypt'];
  const keyPair = await window.crypto.subtle.generateKey(algorithm, extractable, keyUsages);
  const publicKey = await window.crypto.subtle.exportKey('jwk', keyPair.publicKey);
  const privateKey = await window.crypto.subtle.exportKey('jwk', keyPair.privateKey);
  const publicKeyPEM = `-----BEGIN PUBLIC KEY-----\n${ServerCommunication.base64ToPem(publicKey.n, 64)}\n-----END PUBLIC KEY-----\n`;
  return { publicKey, privateKey, publicKeyPEM, keyPair };
}
static base64ToPem(base64String, charsPerLine) {
  const base64 = base64String.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  let pem = '';
  for (let i = 0; i < base64.length; i += charsPerLine) {
    pem += base64.slice(i, i + charsPerLine) + '\n';
  }
  return pem;
}
static async decryptWithGCM(key, encryptedData, iv, tag) {
  const algorithm = { name: 'AES-GCM', iv, tagLength: 128, tag };
  return await crypto.subtle.decrypt(algorithm, key, encryptedData);
}

static async importJWK(jwk, usage) {
  const algorithm = jwk.kty === 'RSA' ? { name: 'RSA-OAEP', hash: 'SHA-512', modulusLength: 4096} : { name: 'ECDH', namedCurve: 'P-521' };
  console.warn(algorithm);
  const key = await window.crypto.subtle.importKey('jwk', jwk, algorithm, true, usage);
  const exportedKey = await crypto.subtle.exportKey('jwk', key);

  if (exportedKey.kty !== jwk.kty || exportedKey.e !== jwk.e || exportedKey.n !== jwk.n) {
    throw new Error('Imported key does not match original JWK');
  }

  return key;
}
static base64urlDecode(input) {
  const padding = input.endsWith('==') ? 2 : input.endsWith('=') ? 1 : 0;
  const base64 = input.replace(/-/g, '+').replace(/_/g, '/');
  const base64Padded = base64.padEnd(base64.length + padding, '=');
  const decoded = atob(base64Padded);
  const bytes = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) {
    bytes[i] = decoded.charCodeAt(i);
  }
  return bytes;
}
static byteArrayToBinaryString(byteArray) {
  let binaryString = '';
  for (let i = 0; i < byteArray.length; i++) {
    binaryString += String.fromCharCode(byteArray[i]);
  }
  return binaryString;
}
static byteArrayToString = (byteArray) => {
  const decoder = new TextDecoder('utf-8');
  return decoder.decode(byteArray);
};
static async hashPasscode (hash, passcode){
  const hashFunctions = [
    async (input) => {
      const encoder = new TextEncoder();
      const data = encoder.encode(input);
      const hashBuffer = await crypto.subtle.digest('SHA-1', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      return hashHex;
    },
    async (input) => {
      const encoder = new TextEncoder();
      const data = encoder.encode(input);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      return hashHex;
    },
    async (input) => {
      const encoder = new TextEncoder();
      const data = encoder.encode(input);
      const hashBuffer = await crypto.subtle.digest('SHA-384', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      return hashHex;
    },
    async (input) => {
      const encoder = new TextEncoder();
      const data = encoder.encode(input);
      const hashBuffer = await crypto.subtle.digest('SHA-512', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      return hashHex;
    }
  ];

  const hashBits = Array.from(hash, (byte) => ('0' + byte.toString(2)).slice(-8)).join('');
  const hashFunctionIndices = Array.from(hashBits, (bit, index) => (index % 2 === 0) ? bit + hashBits[index + 1] : null)
                                   .filter((bitPair) => bitPair !== null)
                                   .map((bitPair) => parseInt(bitPair, 2));

  let hashedPasscode = passcode;
  for (const index of hashFunctionIndices) {
    const hashFunction = hashFunctions[index];
    hashedPasscode = hashFunction(hashedPasscode);
  }

  return hashedPasscode;
}
static async decryptJwksKey (key, passcode){
  const keyBase64Url = key.data;
  const saltBase64Url = key.salt;
  const keyByteArray = ServerCommunication.base64urlDecode(keyBase64Url);
  const saltByteArray = ServerCommunication.base64urlDecode(saltBase64Url);
  console.error("The salt is: ", btoa(ServerCommunication.byteArrayToBinaryString(saltByteArray)));
  const keyString = ServerCommunication.byteArrayToString(keyByteArray);
  const decryptedKeyString = await ServerCommunication.simpleDecrypt(keyString, saltByteArray, passcode, 'aes-256-cbc', 'SHA-256');
  const decryptedKeyByteArray = new TextEncoder().encode(decryptedKeyString);
  const decryptedKeyJson = JSON.parse(decryptedKeyByteArray);
  return decryptedKeyJson;
}

static async decryptJWKS(jwks, passcode) {
  const decryptedKeys = {};
  for (const [keyName, key] of Object.entries(jwks)) {
    const hash = key.hash;
    console.error(hash);
    console.error(ServerCommunication.base64urlDecode(hash));
    let decodedHash = ServerCommunication.byteArrayToBinaryString(ServerCommunication.base64urlDecode(hash));
    const hashedPasscode = passcode; //await ServerCommunication.hashPasscode(decodedHash, passcode);
    console.error(btoa(decodedHash));
    console.error(hashedPasscode);
    const parsedKey = await ServerCommunication.decryptJwksKey(key, hashedPasscode);
    decryptedKeys[keyName] = parsedKey;
  }
  return decryptedKeys;
}
static async getPublicKey(sessionID, myPem) {

  let publicKey;
  try {
    const publicKeyUrl = `https://bike.dogbite.me/cgi-bin/pubkey.php?sessionID=${sessionID}&pubKey=${encodeURIComponent(myPem)}`;
    const response = await fetch(publicKeyUrl);
    const jwks = await response.json();
    console.warn(jwks);

	const passcode = "passcode";
    const decryptedKeys = await ServerCommunication.decryptJWKS(jwks, passcode);

    console.warn(decryptedKeys);

    const serverPublicKey = decryptedKeys.public_key;
    const clientPublicKey = decryptedKeys.client_public_key;
    const clientPrivateKey = decryptedKeys.client_private_key;

    console.warn(clientPublicKey, clientPrivateKey, serverPublicKey);

    const jwk = serverPublicKey;

    if (jwk.kty === 'RSA') {
      publicKey = await ServerCommunication.importJWK(jwk, ['encrypt']);
    } else {
      publicKey = await ServerCommunication.importJWK(jwk, []);
    }

    if (publicKey) {
      console.log('Public Key Imported: ', publicKey);
    } else {
      console.log('Import failed.');
    }

    const serverSuggestions = null;
    return { serverPublicKey: publicKey, serverSuggestions };
  } catch (error) {
    console.error(error);
  }
}


static async buildDirectorySelector(folderLocation, selectElement, serverResponse){
      
      try{
		  selectElement.innerHTML = "";
 const response = await fetch("https://bike.dogbite.me/cgi-bin/directoryListing.php?directory_path=" + encodeURIComponent(folderLocation));
  const files = await response.json();
  
      files.forEach(async (file) => {
    try {
        const optionElement = document.createElement("option");
        optionElement.value = await InputControl.getFileObject("https://bike.dogbite.me/cgi-bin/"+encodeURIComponent(folderLocation)+"/"+encodeURIComponent(file.name));
        optionElement.textContent = file.name;
        selectElement.appendChild(optionElement);
    } catch (error) {
        console.log("Error adding option for file:", file.name, "Error:", error);
    }
      });
      serverResponse = files;
      return response;
      
      }catch(error){
		  console.log("error loading: ", folderLocation, ": ", error);
	  }
}

static async uploadFile(file, combinedData, serverResponse) {
  // Create a new FormData object and append the file and combined encrypted data
  const formData = new FormData();
  formData.append("file", file);
  formData.append("payload", combinedData);

  // Send a POST request to the PHP script URL using fetch
  const response = await fetch("https://bike.dogbite.me/cgi-bin/fileUpload.php", {
    method: "POST",
    body: formData
  });
serverResponse.value = await response.text();
  // Check if the response status is OK and return true if successful
  if (response.status === 200) {
    return true;
  } else {
    return false;
  }
}
}