import type { ZodType, ZodObject, ZodRawShape } from "zod";

/**
 * Minimal Zod-to-JSON-Schema converter.
 * Handles the subset of Zod types used by our tool definitions.
 */
export function zodToJsonSchema(schema: ZodObject<ZodRawShape>): Record<string, any> {
  return convertType(schema);
}

function convertType(schema: ZodType<any>): Record<string, any> {
  const def = (schema as any)._def;

  switch (def.typeName) {
    case "ZodObject": {
      const shape = def.shape();
      const properties: Record<string, any> = {};
      const required: string[] = [];

      for (const [key, value] of Object.entries(shape)) {
        properties[key] = convertType(value as ZodType<any>);
        if (!isOptional(value as ZodType<any>)) {
          required.push(key);
        }
      }

      const result: Record<string, any> = { type: "object", properties };
      if (required.length > 0) {
        result.required = required;
      }
      return result;
    }

    case "ZodString": {
      const result: Record<string, any> = { type: "string" };
      if (def.description) result.description = def.description;
      return result;
    }

    case "ZodNumber": {
      const result: Record<string, any> = { type: "number" };
      if (def.description) result.description = def.description;
      return result;
    }

    case "ZodBoolean": {
      const result: Record<string, any> = { type: "boolean" };
      if (def.description) result.description = def.description;
      return result;
    }

    case "ZodEnum": {
      const result: Record<string, any> = {
        type: "string",
        enum: def.values,
      };
      if (def.description) result.description = def.description;
      return result;
    }

    case "ZodArray": {
      const result: Record<string, any> = {
        type: "array",
        items: convertType(def.type),
      };
      if (def.description) result.description = def.description;
      return result;
    }

    case "ZodOptional": {
      const inner = convertType(def.innerType);
      if (def.description) inner.description = def.description;
      return inner;
    }

    case "ZodDefault": {
      const inner = convertType(def.innerType);
      inner.default = def.defaultValue();
      if (def.description) inner.description = def.description;
      return inner;
    }

    default:
      return { type: "string" };
  }
}

function isOptional(schema: ZodType<any>): boolean {
  const def = (schema as any)._def;
  if (def.typeName === "ZodOptional") return true;
  if (def.typeName === "ZodDefault") return true;
  return false;
}
