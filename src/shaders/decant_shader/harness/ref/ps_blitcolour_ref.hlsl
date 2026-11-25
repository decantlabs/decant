// Shader de reference ECRIT A LA MAIN (aucun contenu du jeu) : meme calcul que PS_BlitColour d'apres son code
// machine — la couleur interpolee multipliee par quatre flottants lus a l'octet 32 du tampon de constantes b27.
// Sert a voir quelle disposition vkd3d-proton donne a un tampon de constantes en parametre racine.
cbuffer Constants : register(b27) { float4 unused0; float4 unused1; float4 factor; };
float4 main(float4 position : SV_Position, float4 color : COLOR0, float2 uv : TEXCOORD0) : SV_Target { return color * factor; }
