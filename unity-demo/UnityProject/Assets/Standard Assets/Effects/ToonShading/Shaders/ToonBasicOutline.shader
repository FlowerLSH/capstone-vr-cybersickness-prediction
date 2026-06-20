Shader "Toon/Basic Outline"
{
    Properties
    {
        _Color ("Main Color", Color) = (1,1,1,1)
        _MainTex ("Base (RGB)", 2D) = "white" {}
        _ToonShade ("Toon Shade", 2D) = "white" {}
        _Outline ("Outline width", Range(0,0.03)) = 0.005
        _OutlineColor ("Outline Color", Color) = (0,0,0,1)
        _Shininess ("Shininess", Range(0.01,1)) = 0.078125
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        Pass
        {
            Name "OUTLINE"
            Cull Front
            ZWrite On
            ColorMask RGB

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            float _Outline;
            fixed4 _OutlineColor;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                v.vertex.xyz += normalize(v.normal) * _Outline;
                o.pos = UnityObjectToClipPos(v.vertex);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return _OutlineColor;
            }
            ENDCG
        }

        Pass
        {
            Name "FORWARD"
            Tags { "LightMode"="ForwardBase" }
            Cull Back

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fwdbase
            #include "UnityCG.cginc"
            #include "Lighting.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _Color;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                fixed3 lighting : COLOR0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);

                fixed3 worldNormal = UnityObjectToWorldNormal(v.normal);
                fixed ndotl = saturate(dot(worldNormal, _WorldSpaceLightPos0.xyz));
                o.lighting = UNITY_LIGHTMODEL_AMBIENT.rgb + (_LightColor0.rgb * (ndotl * 0.5 + 0.5));
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv) * _Color;
                col.rgb *= i.lighting;
                return col;
            }
            ENDCG
        }
    }

    Fallback "Diffuse"
}
