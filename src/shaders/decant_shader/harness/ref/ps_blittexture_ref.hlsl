// Shader de reference ECRIT A LA MAIN (aucun contenu du jeu) : lit une texture t188 avec le sampler s0 a la
// coordonnee TEXCOORD0. Sert a voir quelle disposition vkd3d-proton donne a une texture en table de descripteurs
// et a un sampler statique.
Texture2D    image   : register(t188);
SamplerState sampler0 : register(s0);
float4 main(float4 position : SV_Position, float4 color : COLOR0, float2 uv : TEXCOORD0) : SV_Target { return image.Sample(sampler0, uv); }
