export interface ParamFieldDef {
  key: string
  type: string
  help?: string
  default?: unknown
  choices?: (string | number)[]
}

export interface ParamGroupDef {
  name: string
  fields: ParamFieldDef[]
}
