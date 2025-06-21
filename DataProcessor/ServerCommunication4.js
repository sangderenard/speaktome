class ServerCommunication{


	static async hashPassword(password) {
  const encoder = new TextEncoder();
  const passwordData = encoder.encode(password);

  // Use a null salt
  const saltBuffer = new ArrayBuffer(16);
  const saltView = new Uint8Array(saltBuffer);

  // Import the password as a key
  const importedKey = await crypto.subtle.importKey(
    "raw",
    passwordData,
    { name: "PBKDF2" },
    false,
    ["deriveBits"]
  );

  // Derive bits using the scrypt key derivation function
  const derivedBits = await crypto.subtle.deriveBits(
    {
      name: "PBKDF2",
      salt: saltBuffer,
      iterations: 2000000, // Increase this value to make the hashing more computationally expensive
      hash: "SHA-256",
    },
    importedKey,
    256
  );

  // Convert the ArrayBuffer to a hex string
  const hashedArray = Array.from(new Uint8Array(derivedBits));
  const hashedPassword = hashedArray.map(b => b.toString(16).padStart(2, "0")).join("");

  return hashedPassword;
}
static async buildDirectorySelector(username, password, folder, selectElement, serverResponse) {
  const hashedPassword = await ServerCommunication.hashPassword(password);
  try {
    selectElement.innerHTML = "";
    const response = await fetch(
      "https://bike.dogbite.me/cgi-bin/directoryListing2.php?" +
        new URLSearchParams({
          username: username,
          password: hashedPassword,
          folder: folder,
        })
    );
    const files = await response.json();

    files.forEach(async (file) => {
      try {
        const optionElement = document.createElement("option");
        const fileUrl = "https://bike.dogbite.me/cgi-bin/fileUpload2.php?" +
          new URLSearchParams({
            username: username,
            password: hashedPassword,
            folder: folder,
            file: file.name,
          });
        optionElement.value = fileUrl;
        optionElement.textContent = file.name;
        selectElement.appendChild(optionElement);
      } catch (error) {
        console.log("Error adding option for file:", file.name, "Error:", error);
      }
    });


    serverResponse = files;
    return response;
  } catch (error) {
    console.log("error loading: ", folder, ": ", error);
  }
}
static async uploadFile(file, password, username, folder, serverResponse) {

  const hashedPassword = await ServerCommunication.hashPassword(password);

  // Create a new FormData object and append the file and combined encrypted data
  const formData = new FormData();
  formData.append("file", file);
  formData.append("password", hashedPassword);
  formData.append("username", username);
  formData.append("folder", folder);

  // Send a POST request to the PHP script URL using fetch
  const response = await fetch("https://bike.dogbite.me/cgi-bin/fileUpload2.php", {
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