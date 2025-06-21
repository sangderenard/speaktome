<?php
include_once("key.php");

$sessionID = $_GET["sessionID"];
$clientPubKey = $_GET["pubKey"];

function base64url_encode($data) {
    $urlSafeData = strtr(base64_encode($data), '+/', '-_');
    return rtrim($urlSafeData, '=');
}

function encryptJwk($jwk, $passphrase) {
    $salt = openssl_random_pseudo_bytes(16);
    $hash = openssl_random_pseudo_bytes(16);
    $passcode = hashPasscode($hash, $passphrase);
    error_log("The salt is: " . base64_encode($salt));
    error_log("The hash is: " . base64_encode($hash));
    error_log("The hashed passcode is: " . $passcode);
    $encryptedJwk = simpleEncrypt(json_encode($jwk), $salt, $passcode, 'aes-256-cbc', 'sha256');
    error_log("The encrypted Jwk: " . $encryptedJwk);
    error_log("The encrypted Jwk in base64url: " . base64url_encode($encryptedJwk));
    return array(
        'data' => base64url_encode($encryptedJwk),
        'salt' => base64url_encode($salt),
        'hash' => base64url_encode($hash)
    );
}

function pemToJwk($pem, $keyType) {
    $type = 'public';
    $key = openssl_pkey_get_public($pem);
    if(!$key){
        $type = 'private';
        $key = openssl_pkey_get_private($pem);
        if(!$key){
            return "failure";
        }
    }
    $details = openssl_pkey_get_details($key);
    if ($details['type'] !== OPENSSL_KEYTYPE_RSA) {
        throw new Exception('Invalid key type, expected RSA key');
    }
    $n = base64url_encode($details['rsa']['n']);
    $e = base64url_encode($details['rsa']['e']);
    
    
    $jwk = array(
        'kty' => 'RSA',
        'alg' => 'RSA-OAEP-512',
        'n' => $n,
        'e' => $e,
    );
    
    if ($keyType === 'private') {
        $d = base64url_encode($details['rsa']['d']);
        $p = base64url_encode($details['rsa']['p']);
        $q = base64url_encode($details['rsa']['q']);
        $dp = base64url_encode($details['rsa']['dmp1']);
        $dq = base64url_encode($details['rsa']['dmq1']);
        $qi = base64url_encode($details['rsa']['iqmp']);
        
        $jwk['d'] = $d;
        $jwk['p'] = $p;
        $jwk['q'] = $q;
        $jwk['dp'] = $dp;
        $jwk['dq'] = $dq;
        $jwk['qi'] = $qi;
        $jwk['use'] = 'dec';
    } else {
        $jwk['use'] = 'enc';
    }
    
    return $jwk;
}
testEncryptDecrypt();
// Get the public key contents using key.php
[ $publicKey, $clientPublicKey, $clientPrivateKey ] = generateKeyPair($sessionID, null, null);

$passphrase = "passcode";
$publicKeyJwk = pemToJwk($publicKey, 'public');
$clientPublicKeyJwk = pemToJwk($clientPublicKey, 'public');
$clientPrivateKeyJwk = pemToJwk($clientPrivateKey, 'private');

$encryptedPublicKeyJwk = encryptJwk($publicKeyJwk, $passphrase);
$encryptedClientPublicKeyJwk = encryptJwk($clientPublicKeyJwk, $passphrase);
$encryptedClientPrivateKeyJwk = encryptJwk($clientPrivateKeyJwk, $passphrase);

// Return the encrypted JWKs to the client
header("Content-Type: application/json");
header("Content-Disposition: attachment; filename=\"public_key.jwk\"");
echo json_encode(array(
    'public_key' => $encryptedPublicKeyJwk,
    'client_public_key' => $encryptedClientPublicKeyJwk,
    'client_private_key' => $encryptedClientPrivateKeyJwk
));
?>
