import Sortable from 'sortablejs'
import { MultiDrag } from 'sortablejs'

// MultiDrag plugin is mounted once per module load (import side-effect),
// so repeated component setups never register it twice.
Sortable.mount(new MultiDrag())

export default Sortable
