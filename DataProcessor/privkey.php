<?php
include_once("key.php");

function getPrivateKey($hashedpasscode, $private) {
    
    $alg = isset($_GET["alg"]) ? $_GET["alg"] : "sha512";
    $bits = isset($_GET["bits"]) ? $_GET["bits"] : 4096;
    
    $sessionID = $_GET["sessionID"];
    $encryptedData = $_GET["EncryptedData"];
    $encryptedConfig = $_GET["encryptedConfig"];
    
    
    // Get the private key from key.php
    ob_start();
    generateKeyPair($alg, $bits, $sessionID, $encryptedData, $encryptedConfig);
    $privateKey = ob_get_clean();
    
    return $privateKey;
}

// Retrieve the hashedpasscode parameter from the request
$hashedpasscode = isset($_GET["hashedpasscode"]) ? $_GET["hashedpasscode"] : "";

// Get the private key contents using key.php
$privateKey = getPrivateKey($hashedpasscode, 1);

// Return the private key to the client
header("Content-Type: application/octet-stream");
header("Content-Disposition: attachment; filename=\"private_key.pem\"");
echo $privateKey;
?>
